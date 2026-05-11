"""XGBoost hyperparameter search via 3-fold Q1 time-series CV.

Folds: Q1-2023, Q1-2024, Q1-2025 - season-consistent, no leakage.
If tuned CV MAE improves baseline by >= 0.02%, patches XGB_PARAMS
in train.py automatically.

After this script finishes, run:
    python3 train.py && python3 inference.py

Estimated runtime: 30-90 min on 1 CPU core (20 trials x 3 folds).
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
import xgboost as xgb

from config import SEED, TARGET_COL, DATETIME_COL, REPAIRS_COL, NUM_TURBINES, TURBINE_CAPACITY
from features import make_features
from train import build_direction_speed_avg
from cv_splitter import Q1TimeSeriesSplit

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "dataset_extracted", "dataset", "train_dataset.csv"
)

N_TRIALS = 30
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


def cv_eval(params, df):
    """3-fold Q1 CV, returns mean normalized MAE."""
    splitter = Q1TimeSeriesSplit()
    maes = []
    for df_train, df_val in splitter.split_dataframes(df):
        seasonal_avg = build_seasonal_avg(df_train)
        dsa = build_direction_speed_avg(df_train)
        X_tr = make_features(df_train, seasonal_avg, dsa).values
        X_v = make_features(df_val, seasonal_avg, dsa).values
        avail_tr = get_avail_cap(df_train[REPAIRS_COL])
        avail_v = get_avail_cap(df_val[REPAIRS_COL])
        y_tr_cf = df_train[TARGET_COL].values / avail_tr
        y_v_raw = df_val[TARGET_COL].values
        y_v_cf = y_v_raw / avail_v

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr_cf, eval_set=[(X_v, y_v_cf)], verbose=False)

        preds = clip_preds(model.predict(X_v) * avail_v, df_val[REPAIRS_COL])
        maes.append(normalized_mae(y_v_raw, preds))
    return float(np.mean(maes))


def objective(trial, df):
    params = {
        "objective": "reg:absoluteerror",
        "n_estimators": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0, log=True),
        "seed": SEED,
        "nthread": 1,
        "verbosity": 0,
        "early_stopping_rounds": 100,
        "eval_metric": "mae",
    }
    return cv_eval(params, df)


def print_progress(study, trial):
    elapsed = (trial.datetime_complete - study.trials[0].datetime_start).total_seconds() / 60
    remaining = elapsed / (trial.number + 1) * (N_TRIALS - trial.number - 1)
    print(
        f"  Trial {trial.number + 1:3d}/{N_TRIALS}: {trial.value:.4f}%"
        f" | best: {study.best_value:.4f}%"
        f" | elapsed: {elapsed:.0f} min"
        f" | ETA: {remaining:.0f} min",
        flush=True,
    )


def patch_train_py(best_params):
    """Replace XGB_PARAMS block in train.py with tuned values."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    with open(path, "r") as f:
        src = f.read()

    new_block = (
        "XGB_PARAMS = {\n"
        '    "objective": "reg:absoluteerror",\n'
        '    "n_estimators": 5000,\n'
        f'    "learning_rate": {best_params["learning_rate"]:.5f},\n'
        f'    "max_depth": {best_params["max_depth"]},\n'
        f'    "subsample": {best_params["subsample"]:.4f},\n'
        f'    "colsample_bytree": {best_params["colsample_bytree"]:.4f},\n'
        f'    "min_child_weight": {best_params["min_child_weight"]},\n'
        f'    "reg_alpha": {best_params["reg_alpha"]:.4f},\n'
        f'    "reg_lambda": {best_params["reg_lambda"]:.4f},\n'
        '    "seed": SEED,\n'
        '    "nthread": 1,\n'
        '    "verbosity": 0,\n'
        '    "early_stopping_rounds": 150,\n'
        '    "eval_metric": "mae",\n'
        "}"
    )

    updated = re.sub(
        r'XGB_PARAMS\s*=\s*\{[^}]+\}',
        new_block,
        src,
        flags=re.DOTALL,
    )
    if updated == src:
        print("  Warning: XGB_PARAMS block not found - patch skipped")
        return False
    with open(path, "w") as f:
        f.write(updated)
    return True


def main():
    import sys
    print(f"Python: {sys.version}", flush=True)
    print(f"XGBoost: {xgb.__version__}", flush=True)

    print("Loading data...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    print(f"  Rows: {len(df)}", flush=True)

    print("\nBaseline 3-fold CV (current XGB_PARAMS)...", flush=True)
    baseline_params = {
        "objective": "reg:absoluteerror",
        "n_estimators": 5000,
        "learning_rate": 0.03,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": SEED,
        "nthread": 1,
        "verbosity": 0,
        "early_stopping_rounds": 100,
        "eval_metric": "mae",
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
    with open("models/xgb_best_params.json", "w") as f:
        json.dump(
            {"baseline_cv_mae": baseline_mae, "best_cv_mae": best.value,
             "improvement": improvement, "params": best.params},
            f, indent=2,
        )
    print("\nSaved: models/xgb_best_params.json", flush=True)

    if improvement >= IMPROVEMENT_THRESHOLD:
        print(f"\nImprovement >= {IMPROVEMENT_THRESHOLD}% - patching XGB_PARAMS in train.py...", flush=True)
        ok = patch_train_py(best.params)
        if ok:
            print("  train.py patched.", flush=True)
        print("\nNext step:", flush=True)
        print("  python3 train.py && python3 inference.py", flush=True)
    else:
        print(f"\nImprovement {improvement:.4f}% < {IMPROVEMENT_THRESHOLD}% threshold.", flush=True)
        print("train.py NOT patched. Current XGB params are near-optimal.", flush=True)


if __name__ == "__main__":
    main()
