"""LightGBM hyperparameter search via 3-fold Q1 time-series CV.

Folds: Q1-2023, Q1-2024, Q1-2025 - season-consistent, no leakage
from the Q1-2025 holdout used in previous single-fold LGBM tuning.

If tuned CV MAE improves baseline by >= 0.02%, patches LGBM_PARAMS
in train.py automatically.

After this script finishes, run:
    python3 train.py && python3 inference.py

Estimated runtime: 4-12 hours on 1 CPU core (25 trials x 3 folds).
Progress is printed every trial with best so far.
"""
import os
import json
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import lightgbm as lgb

from config import SEED, TARGET_COL, DATETIME_COL, REPAIRS_COL, NUM_TURBINES, TURBINE_CAPACITY
from features import make_features
from train import build_direction_speed_avg
from cv_splitter import Q1TimeSeriesSplit

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "dataset_extracted", "dataset", "train_dataset.csv"
)

N_TRIALS = 40
CAPACITY = 90.09
IMPROVEMENT_THRESHOLD = 0.02


def normalized_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)) / CAPACITY * 100.0


def get_avail_cap(repairs):
    return (NUM_TURBINES - repairs.values) * TURBINE_CAPACITY


def clip_preds(preds, repairs):
    if isinstance(repairs, pd.Series):
        repairs = repairs.values
    return np.clip(preds, 0.0, (NUM_TURBINES - repairs) * TURBINE_CAPACITY)


def build_seasonal_avg(df):
    return df.groupby(
        [df["month"].astype(int), df["hour_of_day"].astype(int), df[REPAIRS_COL].astype(int)]
    )[TARGET_COL].mean()


def train_lgbm_fold(X_tr, y_tr, X_v, y_v, params):
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_v, label=y_v, reference=dtrain)
    callbacks = [
        lgb.log_evaluation(period=-1),  # silent
        lgb.early_stopping(stopping_rounds=100, verbose=False),
    ]
    model = lgb.train(params, dtrain, valid_sets=[dval], callbacks=callbacks)
    return model


def cv_eval(params, df):
    """3-fold Q1 CV, returns mean normalized MAE."""
    splitter = Q1TimeSeriesSplit()
    maes = []
    for df_train, df_val in splitter.split_dataframes(df):
        seasonal_avg = build_seasonal_avg(df_train)
        dsa = build_direction_speed_avg(df_train)
        X_tr = make_features(df_train, seasonal_avg, dsa)
        X_v = make_features(df_val, seasonal_avg, dsa)
        avail_tr = get_avail_cap(df_train[REPAIRS_COL])
        avail_v = get_avail_cap(df_val[REPAIRS_COL])
        y_tr_cf = df_train[TARGET_COL].values / avail_tr
        y_v_raw = df_val[TARGET_COL].values
        y_v_cf = y_v_raw / avail_v

        model = train_lgbm_fold(X_tr, y_tr_cf, X_v, y_v_cf, params)
        preds = clip_preds(model.predict(X_v) * avail_v, df_val[REPAIRS_COL])
        maes.append(normalized_mae(y_v_raw, preds))
    return float(np.mean(maes))


def objective(trial, df):
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.9),
        "bagging_freq": 1,
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0, log=True),
        "seed": SEED,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "verbose": -1,
    }
    return cv_eval(params, df)


def print_progress(study, trial):
    elapsed = (trial.datetime_complete - study.trials[0].datetime_start).total_seconds() / 60
    print(
        f"  Trial {trial.number + 1:3d}/{N_TRIALS}: {trial.value:.4f}%"
        f" | best: {study.best_value:.4f}%"
        f" | elapsed: {elapsed:.0f} min"
        f" | ETA: {elapsed / (trial.number + 1) * (N_TRIALS - trial.number - 1):.0f} min",
        flush=True,
    )


def patch_train_py(best_params):
    """Replace LGBM_PARAMS block in train.py with tuned values."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    with open(path, "r") as f:
        src = f.read()

    new_block = (
        "# Hyperparameters found via Optuna search (25 trials, 3-fold Q1 CV)\n"
        "LGBM_PARAMS = {\n"
        '    "objective": "regression_l1",\n'
        '    "metric": "mae",\n'
        '    "n_estimators": 5000,\n'
        f'    "learning_rate": {best_params["learning_rate"]:.5f},\n'
        f'    "max_depth": {best_params["max_depth"]},\n'
        f'    "num_leaves": {best_params["num_leaves"]},\n'
        f'    "feature_fraction": {best_params["feature_fraction"]:.4f},\n'
        f'    "bagging_fraction": {best_params["bagging_fraction"]:.4f},\n'
        '    "bagging_freq": 1,\n'
        f'    "min_child_samples": {best_params["min_child_samples"]},\n'
        f'    "reg_alpha": {best_params["reg_alpha"]:.4f},\n'
        f'    "reg_lambda": {best_params["reg_lambda"]:.4f},\n'
        '    "seed": SEED,\n'
        '    "deterministic": True,\n'
        '    "force_row_wise": True,\n'
        '    "num_threads": 1,\n'
        '    "verbose": -1,\n'
        "}"
    )

    updated = re.sub(
        r'(?:# Hyperparameters.*?\n)?LGBM_PARAMS\s*=\s*\{[^}]+\}',
        new_block,
        src,
        flags=re.DOTALL,
    )
    if updated == src:
        print("  Warning: LGBM_PARAMS block not found - patch skipped")
        return False
    with open(path, "w") as f:
        f.write(updated)
    return True


def main():
    import sys
    print(f"Python: {sys.version}", flush=True)
    print(f"LightGBM: {lgb.__version__}", flush=True)

    print("Loading data...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    print(f"  Rows: {len(df)}", flush=True)

    # Current LGBM_PARAMS as baseline
    print("\nBaseline 3-fold CV (current LGBM_PARAMS)...", flush=True)
    baseline_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 5000,
        "learning_rate": 0.03545,
        "max_depth": 4,
        "num_leaves": 87,
        "feature_fraction": 0.4026,
        "bagging_fraction": 0.5826,
        "bagging_freq": 1,
        "min_child_samples": 29,
        "reg_alpha": 0.1076,
        "reg_lambda": 0.2753,
        "seed": SEED,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "verbose": -1,
    }
    baseline_mae = cv_eval(baseline_params, df)
    print(f"  Baseline mean CV MAE: {baseline_mae:.4f}%", flush=True)

    print(f"\nOptuna: {N_TRIALS} trials x 3 folds...", flush=True)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda t: objective(t, df),
        n_trials=N_TRIALS,
        callbacks=[print_progress],
    )

    best = study.best_trial
    improvement = baseline_mae - best.value
    print(f"\n--- Results ---", flush=True)
    print(f"Baseline CV MAE:   {baseline_mae:.4f}%", flush=True)
    print(f"Best tuned CV MAE: {best.value:.4f}%", flush=True)
    print(f"Improvement:       {improvement:+.4f}%", flush=True)
    print("\nBest params:", flush=True)
    for k, v in best.params.items():
        print(f"  {k}: {v}", flush=True)

    os.makedirs("models", exist_ok=True)
    with open("models/lgbm_best_params.json", "w") as f:
        json.dump(
            {"baseline_cv_mae": baseline_mae, "best_cv_mae": best.value,
             "improvement": improvement, "params": best.params},
            f, indent=2,
        )
    print("\nSaved: models/lgbm_best_params.json", flush=True)

    if improvement >= IMPROVEMENT_THRESHOLD:
        print(f"\nImprovement >= {IMPROVEMENT_THRESHOLD}% - patching LGBM_PARAMS in train.py...", flush=True)
        ok = patch_train_py(best.params)
        if ok:
            print("  train.py patched.", flush=True)
        print("\nNext step:", flush=True)
        print("  python3 train.py && python3 inference.py", flush=True)
    else:
        print(f"\nImprovement {improvement:.4f}% < {IMPROVEMENT_THRESHOLD}% threshold.", flush=True)
        print("train.py NOT patched. Current LGBM params are near-optimal.", flush=True)


if __name__ == "__main__":
    main()
