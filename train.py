"""Train LightGBM + XGBoost models for wind power forecasting.

Strategy:
  - LightGBM trains on capacity factor (CF = generation / available_capacity)
  - XGBoost trains on raw MWh generation
  - Diverse targets create better ensemble diversity
  - Both models are retrained on full data after hyperparameter selection via Q1-2025 holdout
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

# Hyperparameters found via Optuna search (25 trials, Q1-2025 holdout)
LGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": 5000,
    "learning_rate": 0.01693,
    "max_depth": 6,
    "num_leaves": 44,
    "feature_fraction": 0.5959,
    "bagging_fraction": 0.5591,
    "bagging_freq": 1,
    "min_child_samples": 23,
    "reg_alpha": 0.4697,
    "reg_lambda": 0.2292,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "verbose": -1,
}

XGB_PARAMS = {
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
    "early_stopping_rounds": 150,
    "eval_metric": "mae",
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


def clip_predictions(preds: np.ndarray, repairs) -> np.ndarray:
    if isinstance(repairs, pd.Series):
        repairs = repairs.values
    max_cap = (NUM_TURBINES - repairs) * TURBINE_CAPACITY
    return np.clip(preds, 0.0, max_cap)


def train_lgbm(X_train, y_train, X_val=None, y_val=None, n_estimators_override=None):
    params = {**LGBM_PARAMS}
    if n_estimators_override:
        params["n_estimators"] = n_estimators_override
    dtrain = lgb.Dataset(X_train, label=y_train)
    callbacks = [lgb.log_evaluation(period=200)]
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
    print("\nStep 1: Training LightGBM on CF target (early stopping Q1-2025)...")
    lgbm_val = train_lgbm(X_train, y_train_cf, X_val, y_val_cf)
    lgbm_best_n = lgbm_val.best_iteration
    lgbm_pred_val = lgbm_val.predict(X_val) * avail_cap_val
    lgbm_pred_val = clip_predictions(lgbm_pred_val, val_repairs)
    mae_lgbm = normalized_mae(y_val, lgbm_pred_val)
    print(f"  Best n_estimators: {lgbm_best_n} | Val MAE: {mae_lgbm:.4f}%")

    print("\nStep 1: Training XGBoost on CF target (early stopping Q1-2025)...")
    xgb_val = train_xgb(X_train.values, y_train_cf, X_val.values, y_val_cf)
    xgb_best_n = xgb_val.best_iteration + 1
    xgb_pred_val = xgb_val.predict(X_val.values) * avail_cap_val
    xgb_pred_val = clip_predictions(xgb_pred_val, val_repairs)
    mae_xgb = normalized_mae(y_val, xgb_pred_val)
    print(f"  Best n_estimators: {xgb_best_n} | Val MAE: {mae_xgb:.4f}%")

    # Weighted ensemble (0.6/0.4 toward better model)
    if mae_lgbm <= mae_xgb:
        w_lgbm, w_xgb = 0.6, 0.4
    else:
        w_lgbm, w_xgb = 0.4, 0.6
    ens_pred = w_lgbm * lgbm_pred_val + w_xgb * xgb_pred_val
    mae_ens_w = normalized_mae(y_val, clip_predictions(ens_pred, val_repairs))
    ens_equal = (lgbm_pred_val + xgb_pred_val) / 2.0
    mae_ens_eq = normalized_mae(y_val, clip_predictions(ens_equal, val_repairs))
    print(f"\n  Ensemble 50/50 val MAE: {mae_ens_eq:.4f}%")
    print(f"  Ensemble weighted val MAE: {mae_ens_w:.4f}%")

    best_val_mae = min(mae_lgbm, mae_xgb, mae_ens_w, mae_ens_eq)
    if best_val_mae == mae_lgbm:
        ensemble_mode = "lgbm"
    elif best_val_mae == mae_xgb:
        ensemble_mode = "xgb"
    elif best_val_mae == mae_ens_eq:
        ensemble_mode = "equal"
    else:
        ensemble_mode = "weighted"
    print(f"  Best mode: {ensemble_mode} | Best val MAE: {best_val_mae:.4f}%")

    # ---- Step 2: retrain on FULL dataset ----
    print("\nStep 2: Retraining on FULL dataset with best n_estimators...")
    seasonal_avg_full = build_seasonal_avg(df)
    X_full = make_features(df, seasonal_avg_full)

    y_full_raw = df[TARGET_COL].values
    avail_cap_full = get_available_cap(df[REPAIRS_COL])
    y_full_cf = y_full_raw / avail_cap_full

    lgbm_n_full = int(lgbm_best_n * 1.1)
    xgb_n_full = int(xgb_best_n * 1.1)
    print(f"  LightGBM n_estimators: {lgbm_n_full}")
    print(f"  XGBoost n_estimators: {xgb_n_full}")

    lgbm_final = train_lgbm(X_full, y_full_cf, n_estimators_override=lgbm_n_full)
    xgb_final = train_xgb(X_full.values, y_full_cf, n_estimators_override=xgb_n_full)

    with open("models/lgbm_model.pkl", "wb") as f:
        pickle.dump(lgbm_final, f)
    with open("models/xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_final, f)

    meta = {
        "ensemble_mode": ensemble_mode,
        "w_lgbm": w_lgbm,
        "w_xgb": w_xgb,
        "mae_lgbm": mae_lgbm,
        "mae_xgb": mae_xgb,
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
