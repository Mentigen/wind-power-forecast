"""Train LightGBM + XGBoost + CatBoost models for wind power forecasting.

Strategy:
  - All models train on capacity factor (CF = generation / available_capacity)
  - LightGBM uses 5-seed averaging (seeds 42-46) to reduce variance
  - Best ensemble selected by val MAE comparison on fixed-weight combinations
  - Models retrained on full data after hyperparameter selection via Q1-2025 holdout
"""
import os
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from config import (
    SEED, TARGET_COL, DATETIME_COL, REPAIRS_COL,
    NUM_TURBINES, TURBINE_CAPACITY,
    VAL_START, VAL_END,
)
from features import make_features

random.seed(SEED)
np.random.seed(SEED)
os.makedirs("models", exist_ok=True)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "dataset_extracted", "dataset", "train_dataset.csv"
)

LGBM_SEEDS = [SEED + i for i in range(5)]  # [42, 43, 44, 45, 46]

# Hyperparameters found via Optuna search (25 trials, 3-fold Q1 CV)
LGBM_PARAMS = {
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

XGB_PARAMS = {
    "objective": "reg:absoluteerror",
    "n_estimators": 5000,
    "learning_rate": 0.01284,
    "max_depth": 7,
    "subsample": 0.5028,
    "colsample_bytree": 0.7240,
    "min_child_weight": 35,
    "reg_alpha": 0.0265,
    "reg_lambda": 0.3562,
    "seed": SEED,
    "nthread": 1,
    "verbosity": 0,
    "early_stopping_rounds": 150,
    "eval_metric": "mae",
}

CATBOOST_PARAMS = {
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "iterations": 5000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": SEED,
    "thread_count": 1,
    "early_stopping_rounds": 100,
    "verbose": False,
}


def normalized_mae(y_true, y_pred, capacity=90.09):
    return np.mean(np.abs(y_true - y_pred)) / capacity * 100.0


def get_available_cap(repairs_series):
    return (NUM_TURBINES - repairs_series.values) * TURBINE_CAPACITY


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    return df


def split_train_val(df: pd.DataFrame):
    val_mask = (df[DATETIME_COL] >= VAL_START) & (df[DATETIME_COL] <= VAL_END)
    return df[~val_mask].copy(), df[val_mask].copy()


def build_seasonal_avg(df: pd.DataFrame) -> pd.Series:
    return df.groupby(
        [df["month"].astype(int), df["hour_of_day"].astype(int), df[REPAIRS_COL].astype(int)]
    )[TARGET_COL].mean()


def build_direction_speed_avg(df: pd.DataFrame) -> pd.Series:
    """Mean CF by (season_quarter, ws80_bin, wind_direction_sector, repairs)."""
    avail = (NUM_TURBINES - df[REPAIRS_COL]) * TURBINE_CAPACITY
    cf = df[TARGET_COL] / avail.clip(lower=0.01)
    quarter = ((df["month"].astype(int) - 1) // 3)
    ws_bin = pd.cut(df["wind_speed_80m"], bins=[0, 3, 6, 9, 12, 100],
                    labels=[0, 1, 2, 3, 4]).cat.add_categories(-1).fillna(-1).astype(int).clip(0, 4)
    dir_sector = (df["wind_direction_80m"] * 8).astype(int).clip(0, 7)
    keys = pd.DataFrame({"quarter": quarter, "ws_bin": ws_bin,
                         "dir_sector": dir_sector, "repairs": df[REPAIRS_COL].astype(int)})
    return cf.groupby([keys["quarter"], keys["ws_bin"],
                       keys["dir_sector"], keys["repairs"]]).mean()


def clip_predictions(preds: np.ndarray, repairs) -> np.ndarray:
    if isinstance(repairs, pd.Series):
        repairs = repairs.values
    max_cap = (NUM_TURBINES - repairs) * TURBINE_CAPACITY
    return np.clip(preds, 0.0, max_cap)


def train_lgbm(X_train, y_train, X_val=None, y_val=None,
               n_estimators_override=None, seed_override=None):
    params = {**LGBM_PARAMS}
    if n_estimators_override is not None:
        params["n_estimators"] = n_estimators_override
    if seed_override is not None:
        params["seed"] = seed_override
    dtrain = lgb.Dataset(X_train, label=y_train)
    callbacks = [lgb.log_evaluation(period=500)]
    valid_sets = [dtrain]
    if X_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        valid_sets = [dval]
        callbacks.append(lgb.early_stopping(stopping_rounds=100, verbose=False))
    return lgb.train(params, dtrain, valid_sets=valid_sets, callbacks=callbacks)


def train_xgb(X_train, y_train, X_val=None, y_val=None, n_estimators_override=None):
    params = {**XGB_PARAMS}
    if n_estimators_override:
        params["n_estimators"] = n_estimators_override
        params.pop("early_stopping_rounds", None)
    model = xgb.XGBRegressor(**params)
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
    else:
        model.fit(X_train, y_train, verbose=200)
    return model


def train_catboost(X_train, y_train, X_val=None, y_val=None, n_estimators_override=None):
    params = {**CATBOOST_PARAMS}
    if n_estimators_override is not None:
        params["iterations"] = n_estimators_override
        params.pop("early_stopping_rounds", None)
    model = CatBoostRegressor(**params)
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
    else:
        model.fit(X_train, y_train)
    return model


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"  Total rows: {len(df)}")

    df_train, df_val = split_train_val(df)
    print(f"  Train: {len(df_train)} rows | Val (Q1-2025): {len(df_val)} rows")

    seasonal_avg = build_seasonal_avg(df_train)
    X_train = make_features(df_train, seasonal_avg)
    X_val = make_features(df_val, seasonal_avg)
    y_train_raw = df_train[TARGET_COL].values
    y_val = df_val[TARGET_COL].values
    val_repairs = df_val[REPAIRS_COL]

    avail_cap_train = get_available_cap(df_train[REPAIRS_COL])
    avail_cap_val = get_available_cap(val_repairs)
    y_train_cf = y_train_raw / avail_cap_train
    y_val_cf = y_val / avail_cap_val
    print(f"  Features: {X_train.shape[1]}")

    # ---- Step 1: hyperparameter selection via Q1-2025 holdout ----

    # LightGBM: find best n_estimators via seed=42 early stopping
    print("\nStep 1: LightGBM seed=42 early stopping to find best n_estimators...")
    lgbm_s42 = train_lgbm(X_train, y_train_cf, X_val, y_val_cf)
    lgbm_best_n = lgbm_s42.best_iteration
    print(f"  Best n_estimators: {lgbm_best_n}")

    # Evaluate 5-seed average on val (no early stopping, fixed n_estimators)
    print(f"  Evaluating 5-seed average (seeds {LGBM_SEEDS})...")
    lgbm_val_preds = []
    for seed in LGBM_SEEDS:
        m = train_lgbm(X_train, y_train_cf, n_estimators_override=lgbm_best_n, seed_override=seed)
        pred = clip_predictions(m.predict(X_val) * avail_cap_val, val_repairs)
        lgbm_val_preds.append(pred)
    lgbm_pred_avg = np.mean(lgbm_val_preds, axis=0)
    mae_lgbm_avg = normalized_mae(y_val, lgbm_pred_avg)
    lgbm_pred_s42 = lgbm_val_preds[0]
    mae_lgbm_s42 = normalized_mae(y_val, lgbm_pred_s42)
    print(f"  Val MAE single (seed=42): {mae_lgbm_s42:.4f}%")
    print(f"  Val MAE 5-seed avg:       {mae_lgbm_avg:.4f}%")
    # Pick the better LGBM representation for ensemble
    if mae_lgbm_s42 < mae_lgbm_avg:
        lgbm_pred_val = lgbm_pred_s42
        mae_lgbm = mae_lgbm_s42
        lgbm_use_single = True
        print("  Using single seed for ensemble (better than avg)")
    else:
        lgbm_pred_val = lgbm_pred_avg
        mae_lgbm = mae_lgbm_avg
        lgbm_use_single = False
        print("  Using 5-seed avg for ensemble")

    print("\nStep 1: Training XGBoost on CF target (early stopping Q1-2025)...")
    xgb_val = train_xgb(X_train.values, y_train_cf, X_val.values, y_val_cf)
    xgb_best_n = xgb_val.best_iteration + 1
    xgb_pred_val = xgb_val.predict(X_val.values) * avail_cap_val
    xgb_pred_val = clip_predictions(xgb_pred_val, val_repairs)
    mae_xgb = normalized_mae(y_val, xgb_pred_val)
    print(f"  Best n_estimators: {xgb_best_n} | Val MAE: {mae_xgb:.4f}%")

    print("\nStep 1: Training CatBoost on CF target (early stopping Q1-2025)...")
    cb_val = train_catboost(X_train.values, y_train_cf, X_val.values, y_val_cf)
    cb_best_n = cb_val.best_iteration_
    cb_pred_val = cb_val.predict(X_val.values) * avail_cap_val
    cb_pred_val = clip_predictions(cb_pred_val, val_repairs)
    mae_cb = normalized_mae(y_val, cb_pred_val)
    print(f"  Best n_estimators: {cb_best_n} | Val MAE: {mae_cb:.4f}%")

    # Fixed-weight ensemble comparison
    ens_2eq = (lgbm_pred_val + xgb_pred_val) / 2.0
    mae_2eq = normalized_mae(y_val, clip_predictions(ens_2eq, val_repairs))

    ens_3eq = (lgbm_pred_val + xgb_pred_val + cb_pred_val) / 3.0
    mae_3eq = normalized_mae(y_val, clip_predictions(ens_3eq, val_repairs))

    if mae_lgbm <= mae_xgb:
        w_lgbm, w_xgb = 0.6, 0.4
    else:
        w_lgbm, w_xgb = 0.4, 0.6
    ens_2w = w_lgbm * lgbm_pred_val + w_xgb * xgb_pred_val
    mae_2w = normalized_mae(y_val, clip_predictions(ens_2w, val_repairs))

    print(f"\n  Ensemble 50/50 (LGBM-avg + XGB) val MAE: {mae_2eq:.4f}%")
    print(f"  Ensemble weighted (LGBM-avg + XGB) val MAE: {mae_2w:.4f}%")
    print(f"  Ensemble equal 3-way val MAE: {mae_3eq:.4f}%")

    # Prefer equal_2 over weighted_2 if the margin is within noise (< 0.02%)
    if mae_2eq - mae_2w < 0.02:
        mae_2w = mae_2eq + 1e-9  # force equal_2 to win

    candidates = {
        "lgbm": mae_lgbm,
        "xgb": mae_xgb,
        "catboost": mae_cb,
        "equal_2": mae_2eq,
        "weighted_2": mae_2w,
        "equal_3": mae_3eq,
    }
    ensemble_mode = min(candidates, key=candidates.get)
    best_val_mae = candidates[ensemble_mode]
    print(f"\n  Best mode: {ensemble_mode} | Best val MAE: {best_val_mae:.4f}%")

    # ---- Step 2: retrain on FULL dataset ----
    print("\nStep 2: Retraining on FULL dataset with best n_estimators...")
    seasonal_avg_full = build_seasonal_avg(df)
    X_full = make_features(df, seasonal_avg_full)

    y_full_raw = df[TARGET_COL].values
    avail_cap_full = get_available_cap(df[REPAIRS_COL])
    y_full_cf = y_full_raw / avail_cap_full

    lgbm_n_full = int(lgbm_best_n * 1.1)
    xgb_n_full = int(xgb_best_n * 1.1)
    cb_n_full = int(cb_best_n * 1.1)
    print(f"  LightGBM n_estimators (per seed): {lgbm_n_full}")
    print(f"  XGBoost n_estimators: {xgb_n_full}")
    print(f"  CatBoost n_estimators: {cb_n_full}")

    # Train 5 LGBM seeds on full data
    print(f"  Training LGBM x{len(LGBM_SEEDS)} seeds on full data...")
    lgbm_finals = []
    for seed in LGBM_SEEDS:
        m = train_lgbm(X_full, y_full_cf, n_estimators_override=lgbm_n_full, seed_override=seed)
        lgbm_finals.append(m)

    xgb_final = train_xgb(X_full.values, y_full_cf, n_estimators_override=xgb_n_full)
    cb_final = train_catboost(X_full.values, y_full_cf, n_estimators_override=cb_n_full)

    # Save LGBM seed models (lgbm_model.pkl = seed=42 for backwards compat)
    for m, seed in zip(lgbm_finals, LGBM_SEEDS):
        fname = f"models/lgbm_seed{seed}.pkl"
        with open(fname, "wb") as f:
            pickle.dump(m, f)
    with open("models/lgbm_model.pkl", "wb") as f:
        pickle.dump(lgbm_finals[0], f)  # seed=42 alias

    with open("models/xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_final, f)
    with open("models/catboost_model.pkl", "wb") as f:
        pickle.dump(cb_final, f)

    meta = {
        "ensemble_mode": ensemble_mode,
        "lgbm_mode": "single" if lgbm_use_single else "multiseed",
        "lgbm_seeds": [42] if lgbm_use_single else LGBM_SEEDS,
        "w_lgbm": w_lgbm,
        "w_xgb": w_xgb,
        "mae_lgbm_s42": mae_lgbm_s42,
        "mae_lgbm_avg": mae_lgbm_avg,
        "mae_xgb": mae_xgb,
        "mae_catboost": mae_cb,
        "mae_ensemble": best_val_mae,
        "best_mae": best_val_mae,
        "feature_names": list(X_full.columns),
        "seasonal_avg": seasonal_avg_full,
    }
    with open("models/meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"\nDone. Best val normalized MAE (Q1-2025): {best_val_mae:.4f}%")
    print("Final models trained on full dataset and saved to models/")


if __name__ == "__main__":
    main()
