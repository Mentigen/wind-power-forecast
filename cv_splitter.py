"""Custom time-series cross-validator for wind power forecasting.

Creates 3 folds where each validation window is Q1 (January-March)
of a given year. This mirrors the actual test period (Q1-2026) so
model selection and hyperparameter tuning generalize to the right season.

    Fold 1: Train = everything before 2023-01-01 | Val = 2023 Q1
    Fold 2: Train = everything before 2024-01-01 | Val = 2024 Q1
    Fold 3: Train = everything before 2025-01-01 | Val = 2025 Q1
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from config import DATETIME_COL


class Q1TimeSeriesSplit(BaseEstimator):
    """Generate train/val index arrays for Q1 cross-validation.

    Parameters
    ----------
    val_years : list of int
        Years to use as validation windows. Default: [2023, 2024, 2025].
    """

    def __init__(self, val_years=None):
        self.val_years = val_years or [2023, 2024, 2025]

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.val_years)

    def split(self, df: pd.DataFrame, y=None, groups=None):
        """Yield (train_idx, val_idx) pairs.

        Parameters
        ----------
        df : DataFrame with a DATETIME_COL column (or DatetimeIndex).
        """
        if isinstance(df.index, pd.DatetimeIndex):
            dt = df.index
        else:
            dt = pd.to_datetime(df[DATETIME_COL])

        for year in self.val_years:
            train_cutoff = pd.Timestamp(f"{year}-01-01")
            val_start = pd.Timestamp(f"{year}-01-01")
            val_end = pd.Timestamp(f"{year}-03-31 23:59:59")

            train_mask = dt < train_cutoff
            val_mask = (dt >= val_start) & (dt <= val_end)

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

            yield train_idx, val_idx

    def split_dataframes(self, df: pd.DataFrame):
        """Yield (df_train, df_val) DataFrame pairs."""
        for train_idx, val_idx in self.split(df):
            yield df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def evaluate_cv(df, model_fn, feature_fn, target_col, capacity=90.09):
    """Run 3-fold Q1 CV and return per-fold and mean normalized MAE.

    Parameters
    ----------
    df : full training DataFrame (sorted ascending by datetime)
    model_fn : callable(X_train, y_train, X_val, y_val) -> model
    feature_fn : callable(df_subset) -> X DataFrame
    target_col : name of target column
    capacity : installed capacity in MW (for metric normalization)

    Returns
    -------
    dict with fold MAEs and mean MAE
    """
    splitter = Q1TimeSeriesSplit()
    results = {}

    for fold, (df_train, df_val) in enumerate(splitter.split_dataframes(df), start=1):
        X_train = feature_fn(df_train)
        X_val = feature_fn(df_val)
        y_train = df_train[target_col].values
        y_val = df_val[target_col].values

        model = model_fn(X_train, y_train, X_val, y_val)
        pred = model.predict(X_val) if hasattr(model, "predict") else model(X_val)
        pred = np.clip(pred, 0, capacity)

        mae = np.mean(np.abs(y_val - pred)) / capacity * 100.0
        results[f"fold{fold}_val_year"] = 2022 + fold
        results[f"fold{fold}_mae"] = mae
        print(f"  Fold {fold} (Q1-{2022+fold}): MAE = {mae:.4f}%")

    maes = [results[k] for k in results if "mae" in k]
    results["mean_mae"] = float(np.mean(maes))
    results["std_mae"] = float(np.std(maes))
    print(f"  Mean MAE: {results['mean_mae']:.4f}% +/- {results['std_mae']:.4f}%")
    return results
