# Wind Power Forecasting

Forecasting hourly electricity generation of a wind power station based on meteorological forecast features.

- Installed capacity: 90.09 MW (26 turbines x 3.465 MW, Siemens Gamesa)
- Hub height: 80 m
- Coordinates: 46.8268N, 38.7179E
- Forecast period: 2026-01-01 - 2026-03-31 (2126 hours)

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9+ required.

---

## Data

Place the hackathon dataset files in `dataset/` (relative to the project root):

```
dataset/
  train_dataset.csv
  valid_features.csv
```

The data is not included in this repository. It is provided separately in the ZIP archive.

---

## Training

```bash
python train.py
```

Trains LightGBM and XGBoost models on capacity factor (CF = generation / available_capacity).
Uses Q1-2025 (January-March 2025) as a holdout set for early stopping.
Saves model weights to `models/`. Prints validation MAE.

Expected output:

```
Loading data...
  Total rows: 32434
  Train: ~30418 rows | Val (Q1-2025): ~2016 rows
  Features: 47
Training LightGBM on CF target (early stopping Q1-2025)...
  Best n_estimators: 1052 | Val MAE: 8.72%
Training XGBoost on CF target (early stopping Q1-2025)...
  Best n_estimators: 647 | Val MAE: 8.74%
  Ensemble 50/50 val MAE: 8.71%
Done. Best val normalized MAE (Q1-2025): 8.71%
```

---

## Inference

```bash
python inference.py
```

Reads `valid_features.csv` and saved model weights, writes `predictions/submission.csv`.

The submission file has:
- No header row
- One column with predicted generation values (MWh)
- 2126 rows, in the same order as `valid_features.csv`
- Decimal separator: dot (e.g., `45.312`)

---

## Approach

### Problem

Supervised regression: predict hourly MWh generation from weather forecast features.

Metric (hackathon): normalized MAE = mean(|predicted - actual|) / 90.09 x 100%.

### Feature Engineering (47 features)

Physics-based features are the core:

**Power curve physics:**
- `ws80_cubed = wind_speed_80m^3` - turbine output scales as v^3 in partial-power region
- `ws120_cubed = wind_speed_120m^3` - 120 m is inside the rotor disk (blade radius ~66 m, hub at 80 m, top of rotor at ~146 m)
- `theoretical_power` - piece-wise power curve approximation (cut-in 3 m/s, rated 12.5 m/s)
- `power_physics = ws80_cubed * available_turbines` - fleet-level wind energy potential

**Turbine availability:**
- `available_turbines = 26 - turbines_under_repair`
- `available_capacity = available_turbines * 3.465 MW`

**Wind shear and turbulence:**
- Shear between 10/80/120/180 m levels
- Shear exponent alpha from log-law fit
- Turbulence proxy: gusts / mean wind speed

**Wind direction** (cyclic sin/cos encoding at 4 heights, direction in [0,1] = fraction of 360 deg)

**Wind components** (u/v at hub height 80 m)

**Time features** (cyclic sin/cos for month and hour of day)

**Air density proxy** (pressure / temperature, ideal gas law)

**Historical seasonal average** (mean generation by month x hour x repair_count, computed from training data only)

**Missing 180 m data:** training data has ~21% missing values at 180 m. Filled by linear extrapolation from 80/120 m; binary `was_imputed_180m` flag added.

### Target Transformation

Both models predict **capacity factor** (CF = generation / available_capacity) rather than raw MWh. This normalizes away the repair-count effect, making the target more consistent across different fleet sizes. At inference, CF prediction is multiplied back by available_capacity.

### Models

1. **LightGBM** - primary model, `objective=regression_l1` (MAE loss). Hyperparameters tuned via Optuna (25 trials). Fully deterministic: `deterministic=True`, `force_row_wise=True`, `num_threads=1`.
2. **XGBoost** - secondary model, `objective=reg:absoluteerror`. Deterministic: `nthread=1`.
3. **CatBoost** - tertiary model, `loss_function=MAE`. Deterministic: `thread_count=1`.
4. **Ensemble** - best option among all single models and fixed-weight combinations, selected by validation MAE.

### Validation

Holdout: Q1-2025 (January-March 2025) - same season as the test period Q1-2026. No data leakage.

After hyperparameter selection, both models are retrained on the full dataset (including Q1-2025) with n_estimators set at `best_iteration * 1.1` to account for the larger training set.

### Post-processing

Predictions are clipped per row to `[0, (26 - repairs_i) * 3.465]`.

### Reproducibility

All random seeds are fixed via `SEED = 42` in `config.py`. Both models use deterministic settings (single-threaded). Running `train.py` and `inference.py` on the same data always produces identical results.

---

## Results

| Model | Val normalized MAE (Q1-2025) |
|-------|------------------------------|
| LightGBM | 8.73% |
| XGBoost | 8.74% |
| CatBoost | 8.85% |
| Ensemble 50/50 (LGBM + XGB) | **8.71%** |

---

## Repository Structure

```
.
├── config.py               # constants and random seed
├── features.py             # feature engineering
├── train.py                # model training
├── inference.py            # prediction generation
├── cv_splitter.py          # Q1 time-series cross-validator (3 folds)
├── ensemble_optimizer.py   # scipy-based ensemble weight optimizer
├── requirements.txt
├── models/                 # saved model weights (generated by train.py)
│   ├── lgbm_model.pkl
│   ├── xgb_model.pkl
│   ├── catboost_model.pkl
│   └── meta.pkl
└── predictions/            # submission CSV (generated by inference.py)
```
