"""Feature ablation: test which feature groups actually help vs 47-feature baseline.

Strategy:
  - Use fixed LightGBM (seed=42, early stopping) as fast proxy
  - Q1-2025 holdout as val set (same as train.py)
  - Test each feature group added beyond original 47 features
  - Report val MAE for each configuration

Run: python3 ablation.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import lightgbm as lgb

# Use local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SEED, TARGET_COL, DATETIME_COL, REPAIRS_COL,
    NUM_TURBINES, TURBINE_CAPACITY, VAL_START, VAL_END,
)
from train import build_seasonal_avg, build_direction_speed_avg, normalized_mae, get_available_cap

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "dataset_extracted", "dataset", "train_dataset.csv"
)

# Same params as in train.py (Optuna-tuned for 47 features)
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

CUT_IN_SPEED = 3.0
RATED_SPEED = 12.5
CUT_OUT_SPEED = 25.0
ROTOR_RADIUS = 66.0
ROTOR_AREA = math.pi * ROTOR_RADIUS ** 2
R_DRY_AIR = 287.05
REF_AIR_DENSITY = 1.225

_SG_CURVE_WS = np.array([
    0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
    6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0,
    11.5, 12.0, 12.5, 25.0, 25.5, 30.0,
], dtype=float)

_SG_CURVE_FRAC = np.array([
    0.000, 0.000, 0.000, 0.000, 0.012, 0.046, 0.101, 0.175, 0.280, 0.397,
    0.519, 0.639, 0.749, 0.841, 0.910, 0.958, 0.984, 0.997, 1.000, 1.000,
    1.000, 1.000, 1.000, 1.000, 0.000, 0.000,
], dtype=float)


def sg_power_fraction(ws):
    return np.interp(ws, _SG_CURVE_WS, _SG_CURVE_FRAC)


def circular_diff_rad(a_frac, b_frac):
    a_rad = a_frac * 2.0 * math.pi
    b_rad = b_frac * 2.0 * math.pi
    diff = a_rad - b_rad
    return np.arctan2(np.sin(diff), np.cos(diff))


def rolling_features(df):
    if DATETIME_COL not in df.columns:
        return pd.DataFrame(index=df.index)
    temp = df[[DATETIME_COL, "wind_speed_80m", "wind_speed_120m",
               "pressure_msl", "wind_gusts_10m", "temperature_80m"]].copy()
    temp[DATETIME_COL] = pd.to_datetime(temp[DATETIME_COL])
    temp = temp.set_index(DATETIME_COL).sort_index()
    full_idx = pd.date_range(temp.index.min(), temp.index.max(), freq="h")
    temp = temp.reindex(full_idx).interpolate(method="linear", limit=6)
    roll_feats = pd.DataFrame(index=full_idx)
    for w in [3, 6, 12, 24]:
        roll_feats[f"ws80_mean_{w}h"] = temp["wind_speed_80m"].rolling(w, min_periods=1).mean()
        roll_feats[f"ws80_max_{w}h"] = temp["wind_speed_80m"].rolling(w, min_periods=1).max()
        roll_feats[f"ws80_std_{w}h"] = temp["wind_speed_80m"].rolling(w, min_periods=1).std().fillna(0)
    for w in [3, 6]:
        roll_feats[f"pressure_diff_{w}h"] = (temp["pressure_msl"] - temp["pressure_msl"].shift(w)).fillna(0)
    roll_feats["ws80_trend_6h"] = (
        temp["wind_speed_80m"] - temp["wind_speed_80m"].rolling(6, min_periods=1).mean()
    )
    roll_feats["gust_trend_6h"] = (
        temp["wind_gusts_10m"] - temp["wind_gusts_10m"].rolling(6, min_periods=1).mean()
    )
    orig_dt = pd.to_datetime(df[DATETIME_COL])
    result = roll_feats.reindex(orig_dt.values)
    result.index = df.index
    return result.fillna(0.0)


def impute_180m(df):
    df = df.copy()
    mask = df["wind_speed_180m"].isna()
    df["was_imputed_180m"] = mask.astype(int)
    df.loc[mask, "wind_speed_180m"] = (
        df.loc[mask, "wind_speed_120m"]
        + 1.5 * (df.loc[mask, "wind_speed_120m"] - df.loc[mask, "wind_speed_80m"])
    )
    df.loc[mask, "wind_direction_180m"] = (
        df.loc[mask, "wind_direction_120m"]
        + 1.5 * (df.loc[mask, "wind_direction_120m"] - df.loc[mask, "wind_direction_80m"])
    )
    df["wind_direction_180m"] = df["wind_direction_180m"].clip(0.0, 1.0)
    return df


def make_features_v1(df, seasonal_avg=None):
    """Original 47-feature version (from git HEAD)."""
    df = impute_180m(df)
    feats = pd.DataFrame(index=df.index)
    repairs = df[REPAIRS_COL]
    available_turbines = NUM_TURBINES - repairs
    ws80 = df["wind_speed_80m"]
    ws120 = df["wind_speed_120m"]

    feats["wind_speed_10m"] = df["wind_speed_10m"]
    feats["wind_speed_80m"] = ws80
    feats["wind_speed_120m"] = ws120
    feats["wind_speed_180m"] = df["wind_speed_180m"]
    feats["wind_gusts_10m"] = df["wind_gusts_10m"]
    feats["temperature_80m"] = df["temperature_80m"]
    feats["temperature_120m"] = df["temperature_120m"]
    feats["pressure_msl"] = df["pressure_msl"]
    feats["rain"] = df["rain"]
    feats["showers"] = df["showers"]
    feats["snowfall"] = df["snowfall"]
    feats["cloud_cover_low"] = df["cloud_cover_low"]
    feats["was_imputed_180m"] = df["was_imputed_180m"]
    feats["available_turbines"] = available_turbines
    feats["available_capacity"] = available_turbines * TURBINE_CAPACITY
    feats["ws80_sq"] = ws80 ** 2
    feats["ws80_cubed"] = ws80 ** 3
    feats["ws120_cubed"] = ws120 ** 3

    # Simple cubic power curve approximation
    ratio = ((ws80 - CUT_IN_SPEED) / (RATED_SPEED - CUT_IN_SPEED)).clip(0.0, 1.0)
    feats["theoretical_power"] = (ratio ** 3 * available_turbines * TURBINE_CAPACITY).clip(0.0, available_turbines * TURBINE_CAPACITY)

    feats["below_cutin"] = (ws80 < CUT_IN_SPEED).astype(int)
    feats["partial_power"] = ((ws80 >= CUT_IN_SPEED) & (ws80 < RATED_SPEED)).astype(int)
    feats["at_rated"] = (ws80 >= RATED_SPEED).astype(int)

    feats["wind_shear_80_10"] = ws80 - df["wind_speed_10m"]
    feats["wind_shear_120_80"] = ws120 - ws80
    feats["wind_shear_180_80"] = df["wind_speed_180m"] - ws80

    ws10_safe = df["wind_speed_10m"].clip(lower=0.01)
    ws80_safe = ws80.clip(lower=0.01)
    feats["shear_exponent"] = np.log(ws80_safe / ws10_safe) / math.log(80 / 10)

    for height in ["10m", "80m", "120m", "180m"]:
        col = f"wind_direction_{height}"
        angle = df[col] * 2.0 * math.pi
        feats[f"wd_{height}_sin"] = np.sin(angle)
        feats[f"wd_{height}_cos"] = np.cos(angle)

    angle_80 = df["wind_direction_80m"] * 2.0 * math.pi
    feats["u_component_80m"] = ws80 * np.sin(angle_80)
    feats["v_component_80m"] = ws80 * np.cos(angle_80)

    feats["turbulence"] = df["wind_gusts_10m"] / (df["wind_speed_10m"] + 0.1)

    month = df["month"].astype(float)
    hour = df["hour_of_day"].astype(float)
    feats["month_sin"] = np.sin(2.0 * math.pi * month / 12.0)
    feats["month_cos"] = np.cos(2.0 * math.pi * month / 12.0)
    feats["hour_sin"] = np.sin(2.0 * math.pi * hour / 24.0)
    feats["hour_cos"] = np.cos(2.0 * math.pi * hour / 24.0)
    feats["month"] = month
    feats["hour_of_day"] = hour

    feats["temp_delta_80_120"] = df["temperature_80m"] - df["temperature_120m"]
    feats["air_density_proxy"] = df["pressure_msl"] / (df["temperature_80m"] + 273.15)
    feats["power_physics"] = feats["ws80_cubed"] * available_turbines

    if seasonal_avg is not None:
        key = pd.MultiIndex.from_arrays([
            df["month"].astype(int), df["hour_of_day"].astype(int), repairs.astype(int),
        ])
        lookup = seasonal_avg.reindex(key)
        lookup.index = df.index
        feats["seasonal_avg"] = lookup.fillna(seasonal_avg.mean()).values

    if DATETIME_COL in df.columns:
        roll = rolling_features(df)
        feats = pd.concat([feats, roll], axis=1)

    return feats


def make_features_with_groups(df, seasonal_avg=None, dsa=None,
                               add_sg_curve=True, add_air_density=True,
                               add_wind_veer=True, add_rotor_eff=True,
                               add_dsa=True, add_uv_120m=True,
                               add_near_cutout=True):
    """V1 base + optional feature groups for ablation."""
    feats = make_features_v1(df, seasonal_avg)
    df = impute_180m(df)
    ws80 = df["wind_speed_80m"]
    ws120 = df["wind_speed_120m"]
    repairs = df[REPAIRS_COL]
    available_turbines = NUM_TURBINES - repairs

    if add_sg_curve:
        sg_frac = sg_power_fraction(ws80.values)
        feats["sg_power_output"] = sg_frac * TURBINE_CAPACITY * available_turbines
        feats["sg_power_frac"] = sg_frac

    if add_air_density:
        t_kelvin = df["temperature_80m"] + 273.15
        p_pascal = df["pressure_msl"] * 100.0
        air_density = p_pascal / (R_DRY_AIR * t_kelvin)
        feats["air_density"] = air_density
        density_ratio = air_density / REF_AIR_DENSITY
        feats["density_ratio"] = density_ratio
        if add_sg_curve:
            feats["sg_power_density_corrected"] = feats["sg_power_output"] * density_ratio
        else:
            sg_frac = sg_power_fraction(ws80.values)
            feats["sg_power_density_corrected"] = sg_frac * TURBINE_CAPACITY * available_turbines * density_ratio
        feats["kinetic_power_80m"] = 0.5 * air_density * ROTOR_AREA * (ws80 ** 3) * available_turbines / 1e6

    if add_wind_veer:
        feats["wind_veer_10_80"] = circular_diff_rad(df["wind_direction_80m"], df["wind_direction_10m"])
        feats["wind_veer_80_120"] = circular_diff_rad(df["wind_direction_120m"], df["wind_direction_80m"])
        feats["wind_veer_80_180"] = circular_diff_rad(df["wind_direction_180m"], df["wind_direction_80m"])

    if add_rotor_eff:
        alpha = feats["shear_exponent"].clip(-0.5, 1.0).values
        h_bot, h_hub, h_top = 14.0, 80.0, 146.0
        rotor_h = h_top - h_bot
        a3 = 3.0 * alpha + 1.0
        safe_a3 = np.where(np.abs(a3) > 0.05, a3, 0.05)
        integral = (h_hub ** (-3 * alpha)) * (h_top ** a3 - h_bot ** a3) / safe_a3
        ws80_safe2 = ws80.values.clip(0.01)
        ws_eff3 = ws80_safe2 ** 3 * integral / rotor_h
        feats["ws_rotor_eff"] = np.where(ws_eff3 > 0, ws_eff3 ** (1.0 / 3.0), 0.0)
        feats["ws_rotor_eff_cubed"] = np.maximum(ws_eff3, 0.0)

    if add_uv_120m:
        angle_120 = df["wind_direction_120m"] * 2.0 * math.pi
        feats["u_component_120m"] = ws120 * np.sin(angle_120)
        feats["v_component_120m"] = ws120 * np.cos(angle_120)

    if add_near_cutout:
        feats["near_cutout"] = (ws80 >= 20.0).astype(int)

    if add_dsa and dsa is not None:
        quarter = ((df["month"].astype(int) - 1) // 3)
        ws_bin = pd.cut(ws80, bins=[0, 3, 6, 9, 12, 100],
                        labels=[0, 1, 2, 3, 4]).cat.add_categories(-1).fillna(-1).astype(int).clip(0, 4)
        dir_sector = (df["wind_direction_80m"] * 8).astype(int).clip(0, 7)
        ds_key = pd.MultiIndex.from_arrays([quarter, ws_bin, dir_sector, repairs.astype(int)])
        ds_lookup = dsa.reindex(ds_key)
        ds_lookup.index = df.index
        feats["direction_speed_avg"] = ds_lookup.fillna(dsa.mean()).values

    return feats


def eval_config(name, df_train, df_val, make_fn, **make_kwargs):
    """Train single LGBM seed=42 with early stopping, return val MAE."""
    seasonal_avg = build_seasonal_avg(df_train)
    dsa = build_direction_speed_avg(df_train) if make_kwargs.get("add_dsa") or make_kwargs.get("dsa") is not None else None
    make_kwargs["seasonal_avg"] = seasonal_avg
    make_kwargs["dsa"] = dsa

    X_tr = make_fn(df_train, **make_kwargs)
    X_v = make_fn(df_val, **make_kwargs)

    avail_tr = get_available_cap(df_train[REPAIRS_COL])
    avail_v = get_available_cap(df_val[REPAIRS_COL])
    y_tr_cf = df_train[TARGET_COL].values / avail_tr
    y_v_raw = df_val[TARGET_COL].values
    y_v_cf = y_v_raw / avail_v
    val_repairs = df_val[REPAIRS_COL]

    dtrain = lgb.Dataset(X_tr, label=y_tr_cf)
    dval = lgb.Dataset(X_v, label=y_v_cf, reference=dtrain)
    callbacks = [
        lgb.log_evaluation(period=-1),
        lgb.early_stopping(stopping_rounds=100, verbose=False),
    ]
    model = lgb.train(LGBM_PARAMS, dtrain, valid_sets=[dval], callbacks=callbacks)

    preds_cf = model.predict(X_v)
    preds = np.clip(preds_cf * avail_v, 0.0, (NUM_TURBINES - val_repairs.values) * TURBINE_CAPACITY)
    mae = normalized_mae(y_v_raw, preds)
    n_trees = model.best_iteration
    print(f"  {name:<40s}: MAE={mae:.4f}%  trees={n_trees}  feats={X_tr.shape[1]}")
    return mae


def main():
    print("Loading data...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)

    val_mask = (df[DATETIME_COL] >= VAL_START) & (df[DATETIME_COL] <= VAL_END)
    df_train = df[~val_mask].copy()
    df_val = df[val_mask].copy()
    print(f"  Train: {len(df_train)}  Val: {len(df_val)}", flush=True)

    print("\n--- Ablation (LGBM seed=42, early stopping) ---", flush=True)

    # Baseline V1: 47 features
    baseline = eval_config(
        "V1 baseline (47 feats)",
        df_train, df_val,
        lambda df, **kw: make_features_v1(df, kw.get("seasonal_avg")),
    )

    # Each group added one at a time on top of V1
    eval_config(
        "+SG power curve",
        df_train, df_val,
        make_features_with_groups,
        add_sg_curve=True, add_air_density=False, add_wind_veer=False,
        add_rotor_eff=False, add_dsa=False, add_uv_120m=False, add_near_cutout=False,
    )
    eval_config(
        "+Air density (exact)",
        df_train, df_val,
        make_features_with_groups,
        add_sg_curve=False, add_air_density=True, add_wind_veer=False,
        add_rotor_eff=False, add_dsa=False, add_uv_120m=False, add_near_cutout=False,
    )
    eval_config(
        "+Wind veer (3 levels)",
        df_train, df_val,
        make_features_with_groups,
        add_sg_curve=False, add_air_density=False, add_wind_veer=True,
        add_rotor_eff=False, add_dsa=False, add_uv_120m=False, add_near_cutout=False,
    )
    eval_config(
        "+Rotor effective wind speed",
        df_train, df_val,
        make_features_with_groups,
        add_sg_curve=False, add_air_density=False, add_wind_veer=False,
        add_rotor_eff=True, add_dsa=False, add_uv_120m=False, add_near_cutout=False,
    )
    eval_config(
        "+Direction-speed-avg lookup",
        df_train, df_val,
        make_features_with_groups,
        add_sg_curve=False, add_air_density=False, add_wind_veer=False,
        add_rotor_eff=False, add_dsa=True, add_uv_120m=False, add_near_cutout=False,
    )
    eval_config(
        "+UV 120m components",
        df_train, df_val,
        make_features_with_groups,
        add_sg_curve=False, add_air_density=False, add_wind_veer=False,
        add_rotor_eff=False, add_dsa=False, add_uv_120m=True, add_near_cutout=False,
    )
    eval_config(
        "+Near cutout indicator",
        df_train, df_val,
        make_features_with_groups,
        add_sg_curve=False, add_air_density=False, add_wind_veer=False,
        add_rotor_eff=False, add_dsa=False, add_uv_120m=False, add_near_cutout=True,
    )

    # Best combinations
    print("\n--- Combinations ---", flush=True)
    eval_config(
        "ALL new features (78)",
        df_train, df_val,
        make_features_with_groups,
        add_sg_curve=True, add_air_density=True, add_wind_veer=True,
        add_rotor_eff=True, add_dsa=True, add_uv_120m=True, add_near_cutout=True,
    )
    eval_config(
        "SG + AirDensity + DSA",
        df_train, df_val,
        make_features_with_groups,
        add_sg_curve=True, add_air_density=True, add_wind_veer=False,
        add_rotor_eff=False, add_dsa=True, add_uv_120m=False, add_near_cutout=False,
    )
    eval_config(
        "SG + AirDensity + WindVeer + DSA",
        df_train, df_val,
        make_features_with_groups,
        add_sg_curve=True, add_air_density=True, add_wind_veer=True,
        add_rotor_eff=False, add_dsa=True, add_uv_120m=False, add_near_cutout=False,
    )

    print(f"\n  Baseline: {baseline:.4f}%", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
