"""Generate predictions for the test period from saved model weights."""
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import DATETIME_COL, REPAIRS_COL, NUM_TURBINES, TURBINE_CAPACITY
from features import make_features

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "dataset_extracted", "dataset", "valid_features.csv"
)
OUTPUT_PATH = "predictions/submission.csv"


def clip_predictions(preds: np.ndarray, repairs: pd.Series) -> np.ndarray:
    max_cap = (NUM_TURBINES - repairs.values) * TURBINE_CAPACITY
    return np.clip(preds, 0.0, max_cap)


def main():
    os.makedirs("predictions", exist_ok=True)

    print("Loading valid features...")
    df_valid = pd.read_csv(DATA_PATH, parse_dates=[DATETIME_COL])
    print(f"  Rows: {len(df_valid)}")
    print(f"  Date range: {df_valid[DATETIME_COL].min()} - {df_valid[DATETIME_COL].max()}")

    print("Loading models...")
    with open("models/lgbm_model.pkl", "rb") as f:
        lgbm_model = pickle.load(f)
    with open("models/xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("models/meta.pkl", "rb") as f:
        meta = pickle.load(f)

    seasonal_avg = meta.get("seasonal_avg")
    ensemble_mode = meta.get("ensemble_mode", "lgbm")
    w_lgbm = meta.get("w_lgbm", 0.5)
    w_xgb = meta.get("w_xgb", 0.5)
    print(f"  Best val MAE (Q1-2025): {meta['best_mae']:.4f}%")
    print(f"  Ensemble mode: {ensemble_mode}")

    # Build features preserving original row order (descending date = grader alignment)
    X_valid = make_features(df_valid, seasonal_avg)
    valid_repairs = df_valid[REPAIRS_COL]
    avail_cap = (NUM_TURBINES - valid_repairs.values) * TURBINE_CAPACITY

    lgbm_pred = lgbm_model.predict(X_valid) * avail_cap
    xgb_pred = xgb_model.predict(X_valid.values) * avail_cap

    # Load CatBoost if ensemble mode requires it
    cb_pred = None
    if ensemble_mode in ("catboost", "equal_3"):
        cb_path = "models/catboost_model.pkl"
        if os.path.exists(cb_path):
            with open(cb_path, "rb") as f:
                cb_model = pickle.load(f)
            cb_pred = cb_model.predict(X_valid.values) * avail_cap
        else:
            print("  Warning: catboost_model.pkl not found, falling back to equal_2")
            ensemble_mode = "equal_2"

    if ensemble_mode == "equal_3" and cb_pred is not None:
        preds = (lgbm_pred + xgb_pred + cb_pred) / 3.0
    elif ensemble_mode == "catboost" and cb_pred is not None:
        preds = cb_pred
    elif ensemble_mode == "equal_2":
        preds = (lgbm_pred + xgb_pred) / 2.0
    elif ensemble_mode == "weighted_2":
        preds = w_lgbm * lgbm_pred + w_xgb * xgb_pred
    elif ensemble_mode == "xgb":
        preds = xgb_pred
    else:  # lgbm or fallback
        preds = lgbm_pred

    # Hard clip: no negatives, no values above per-row available capacity
    preds = clip_predictions(preds, valid_repairs)

    # Write submission: no header, one column, same row order as valid_features.csv
    pd.Series(preds).to_csv(OUTPUT_PATH, index=False, header=False)

    print(f"\nSubmission saved to {OUTPUT_PATH}")
    print(f"  Rows: {len(preds)}")
    print(f"  Min: {preds.min():.3f}  Max: {preds.max():.3f}  Mean: {preds.mean():.3f}")
    print(f"  First 5: {np.round(preds[:5], 3).tolist()}")


if __name__ == "__main__":
    main()
