"""CatBoost hyperparameter search via 3-fold Q1 time-series CV.

Folds: Q1-2023, Q1-2024, Q1-2025 - season-consistent, no leakage.
If tuned CV MAE improves baseline by >= 0.02%, patches CATBOOST_PARAMS
in train.py automatically.

After this script finishes, run:
    python3 train.py && python3 inference.py

Estimated runtime: 30-90 min depending on CPU speed.
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

from catboost import CatBoostRegressor

from config import SEED, TARGET_COL, DATETIME_COL, REPAIRS_COL, NUM_TURBINES, TURBINE_CAPACITY
from features import make_features
from cv_splitter import Q1TimeSeriesSplit

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "dataset_extracted", "dataset", "train_dataset.csv"
)

N_TRIALS = 30
CAPACITY = 90.09
IMPROVEMENT_THRESHOLD = 0.02  # patch train.py only if improvement >= 0.02%


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
        X_tr = make_features(df_train, seasonal_avg).values
        X_v = make_features(df_val, seasonal_avg).values
        avail_tr = get_avail_cap(df_train[REPAIRS_COL])
        avail_v = get_avail_cap(df_val[REPAIRS_COL])
        y_tr_cf = df_train[TARGET_COL].values / avail_tr
        y_v_raw = df_val[TARGET_COL].values
        y_v_cf = y_v_raw / avail_v

        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr_cf, eval_set=(X_v, y_v_cf))

        preds = clip_preds(model.predict(X_v) * avail_v, df_val[REPAIRS_COL])
        maes.append(normalized_mae(y_v_raw, preds))
    return float(np.mean(maes))


def objective(trial, df):
    params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "random_strength": trial.suggest_float("random_strength", 0.1, 2.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_seed": SEED,
        "thread_count": -1,          # all cores during search
        "early_stopping_rounds": 100,
        "verbose": False,
    }
    return cv_eval(params, df)


def print_progress(study, trial):
    if trial.number % 5 == 0 or trial.number == 0:
        print(f"  Trial {trial.number:3d}: {trial.value:.4f}% | best: {study.best_value:.4f}%")


def patch_train_py(best_params):
    """Replace CATBOOST_PARAMS block in train.py with tuned values."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    with open(path, "r") as f:
        src = f.read()

    new_block = (
        "CATBOOST_PARAMS = {\n"
        '    "loss_function": "MAE",\n'
        '    "eval_metric": "MAE",\n'
        '    "iterations": 5000,\n'
        f'    "learning_rate": {best_params["learning_rate"]:.5f},\n'
        f'    "depth": {best_params["depth"]},\n'
        f'    "l2_leaf_reg": {best_params["l2_leaf_reg"]:.4f},\n'
        f'    "bagging_temperature": {best_params["bagging_temperature"]:.4f},\n'
        f'    "random_strength": {best_params["random_strength"]:.4f},\n'
        f'    "border_count": {best_params["border_count"]},\n'
        '    "random_seed": SEED,\n'
        '    "thread_count": 1,\n'
        '    "early_stopping_rounds": 100,\n'
        '    "verbose": False,\n'
        "}"
    )

    # Match the existing CATBOOST_PARAMS = { ... } block (no nested braces inside)
    updated = re.sub(
        r'CATBOOST_PARAMS\s*=\s*\{[^}]+\}',
        new_block,
        src,
        flags=re.DOTALL,
    )
    if updated == src:
        print("  Warning: CATBOOST_PARAMS block not found in train.py - patch skipped")
        return False
    with open(path, "w") as f:
        f.write(updated)
    return True


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    print(f"  Rows: {len(df)}")

    # Baseline: current default CatBoost params evaluated over 3 folds
    print("\nBaseline 3-fold CV (default CatBoost params)...")
    baseline_params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 5000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": SEED,
        "thread_count": -1,
        "early_stopping_rounds": 100,
        "verbose": False,
    }
    baseline_mae = cv_eval(baseline_params, df)
    print(f"  Baseline mean CV MAE: {baseline_mae:.4f}%")

    print(f"\nOptuna: {N_TRIALS} trials x 3 folds (thread_count=-1)...")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda t: objective(t, df),
        n_trials=N_TRIALS,
        callbacks=[print_progress],
    )

    best = study.best_trial
    improvement = baseline_mae - best.value
    print(f"\n--- Results ---")
    print(f"Baseline CV MAE:   {baseline_mae:.4f}%")
    print(f"Best tuned CV MAE: {best.value:.4f}%")
    print(f"Improvement:       {improvement:+.4f}%")
    print("\nBest params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    os.makedirs("models", exist_ok=True)
    with open("models/catboost_best_params.json", "w") as f:
        json.dump(
            {"baseline_cv_mae": baseline_mae, "best_cv_mae": best.value,
             "improvement": improvement, "params": best.params},
            f, indent=2,
        )
    print("\nSaved: models/catboost_best_params.json")

    if improvement >= IMPROVEMENT_THRESHOLD:
        print(f"\nImprovement >= {IMPROVEMENT_THRESHOLD}% - patching CATBOOST_PARAMS in train.py...")
        ok = patch_train_py(best.params)
        if ok:
            print("  train.py patched.")
        print("\nNext step:")
        print("  python3 train.py && python3 inference.py")
    else:
        print(f"\nImprovement {improvement:.4f}% < {IMPROVEMENT_THRESHOLD}% threshold.")
        print("train.py NOT patched (marginal gain not worth the change).")
        print("Current solution is already near-optimal for CatBoost.")


if __name__ == "__main__":
    main()
