import math
import pandas as pd
import numpy as np
from config import NUM_TURBINES, TURBINE_CAPACITY, REPAIRS_COL, DATETIME_COL


CUT_IN_SPEED = 3.0      # m/s
RATED_SPEED = 12.5      # m/s


def _impute_180m(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing 180m wind data by linear extrapolation from 80m and 120m."""
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


def _rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling window statistics on meteorological forecast data.

    The data may have gaps, so we resample to a complete hourly grid,
    compute rolling stats, then join back on the original timestamps.
    """
    if DATETIME_COL not in df.columns:
        return pd.DataFrame(index=df.index)

    dt = pd.to_datetime(df[DATETIME_COL])
    # Work in sorted chronological order on a complete hourly grid
    temp = df[[DATETIME_COL, "wind_speed_80m", "wind_speed_120m",
               "pressure_msl", "wind_gusts_10m", "temperature_80m"]].copy()
    temp[DATETIME_COL] = pd.to_datetime(temp[DATETIME_COL])
    temp = temp.set_index(DATETIME_COL).sort_index()

    # Resample to complete hourly grid (forward-fill short gaps for rolling)
    full_idx = pd.date_range(temp.index.min(), temp.index.max(), freq="h")
    temp = temp.reindex(full_idx).interpolate(method="linear", limit=6)

    roll_feats = pd.DataFrame(index=full_idx)
    for w in [3, 6, 12, 24]:
        roll_feats[f"ws80_mean_{w}h"] = temp["wind_speed_80m"].rolling(w, min_periods=1).mean()
        roll_feats[f"ws80_max_{w}h"] = temp["wind_speed_80m"].rolling(w, min_periods=1).max()
        roll_feats[f"ws80_std_{w}h"] = temp["wind_speed_80m"].rolling(w, min_periods=1).std().fillna(0)
    for w in [3, 6]:
        roll_feats[f"pressure_diff_{w}h"] = (
            temp["pressure_msl"] - temp["pressure_msl"].shift(w)
        ).fillna(0)
    roll_feats["ws80_trend_6h"] = (
        temp["wind_speed_80m"] - temp["wind_speed_80m"].rolling(6, min_periods=1).mean()
    )
    roll_feats["gust_trend_6h"] = (
        temp["wind_gusts_10m"] - temp["wind_gusts_10m"].rolling(6, min_periods=1).mean()
    )

    # Join back on original timestamps
    orig_dt = pd.to_datetime(df[DATETIME_COL])
    result = roll_feats.reindex(orig_dt.values)
    result.index = df.index
    return result


def _power_curve_output(ws: pd.Series, turbines: pd.Series) -> pd.Series:
    single_rated = TURBINE_CAPACITY
    cap = turbines * single_rated
    ratio = ((ws - CUT_IN_SPEED) / (RATED_SPEED - CUT_IN_SPEED)).clip(0.0, 1.0)
    power_fraction = ratio ** 3
    return (power_fraction * cap).clip(0.0, cap)


def make_features(df: pd.DataFrame, seasonal_avg: pd.Series = None) -> pd.DataFrame:
    df = _impute_180m(df)
    feats = pd.DataFrame(index=df.index)

    repairs = df[REPAIRS_COL]
    available_turbines = NUM_TURBINES - repairs
    ws80 = df["wind_speed_80m"]
    ws120 = df["wind_speed_120m"]

    # ---- raw meteorological features ----
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

    # ---- turbine availability ----
    feats["available_turbines"] = available_turbines
    feats["available_capacity"] = available_turbines * TURBINE_CAPACITY

    # ---- power curve physics: P ~ v^3 at hub height (80m) ----
    feats["ws80_sq"] = ws80 ** 2
    feats["ws80_cubed"] = ws80 ** 3
    feats["ws120_cubed"] = ws120 ** 3

    # ---- theoretical power curve output ----
    feats["theoretical_power"] = _power_curve_output(ws80, available_turbines)

    # ---- power potential scaled by available turbines ----
    feats["power_physics"] = feats["ws80_cubed"] * available_turbines

    # ---- power curve zone indicators ----
    feats["below_cutin"] = (ws80 < CUT_IN_SPEED).astype(int)
    feats["partial_power"] = ((ws80 >= CUT_IN_SPEED) & (ws80 < RATED_SPEED)).astype(int)
    feats["at_rated"] = (ws80 >= RATED_SPEED).astype(int)

    # ---- wind shear (vertical gradient) ----
    feats["wind_shear_80_10"] = ws80 - df["wind_speed_10m"]
    feats["wind_shear_120_80"] = ws120 - ws80
    feats["wind_shear_180_80"] = df["wind_speed_180m"] - ws80

    # ---- shear exponent alpha = log(v2/v1) / log(h2/h1) ----
    ws10_safe = df["wind_speed_10m"].clip(lower=0.01)
    ws80_safe = ws80.clip(lower=0.01)
    feats["shear_exponent"] = np.log(ws80_safe / ws10_safe) / math.log(80 / 10)

    # ---- turbulence proxy ----
    feats["turbulence"] = df["wind_gusts_10m"] / (df["wind_speed_10m"] + 0.1)

    # ---- wind direction: cyclic encoding (direction in [0,1] = fraction of 360deg) ----
    for height in ["10m", "80m", "120m", "180m"]:
        col = f"wind_direction_{height}"
        angle = df[col] * 2.0 * math.pi
        feats[f"wd_{height}_sin"] = np.sin(angle)
        feats[f"wd_{height}_cos"] = np.cos(angle)

    # ---- u/v wind components at hub height ----
    angle_80 = df["wind_direction_80m"] * 2.0 * math.pi
    feats["u_component_80m"] = ws80 * np.sin(angle_80)
    feats["v_component_80m"] = ws80 * np.cos(angle_80)

    # ---- time: cyclic encoding ----
    month = df["month"].astype(float)
    hour = df["hour_of_day"].astype(float)
    feats["month_sin"] = np.sin(2.0 * math.pi * month / 12.0)
    feats["month_cos"] = np.cos(2.0 * math.pi * month / 12.0)
    feats["hour_sin"] = np.sin(2.0 * math.pi * hour / 24.0)
    feats["hour_cos"] = np.cos(2.0 * math.pi * hour / 24.0)
    feats["month"] = month
    feats["hour_of_day"] = hour

    # ---- air density proxy ----
    feats["temp_delta_80_120"] = df["temperature_80m"] - df["temperature_120m"]
    feats["air_density_proxy"] = df["pressure_msl"] / (df["temperature_80m"] + 273.15)

    # ---- historical seasonal average (lookup by month x hour x repairs) ----
    if seasonal_avg is not None:
        key = pd.MultiIndex.from_arrays([
            df["month"].astype(int),
            df["hour_of_day"].astype(int),
            repairs.astype(int),
        ])
        lookup = seasonal_avg.reindex(key)
        lookup.index = df.index
        feats["seasonal_avg"] = lookup.fillna(seasonal_avg.mean()).values

    return feats
