SEED = 42

INSTALLED_CAPACITY = 90.09   # MW
NUM_TURBINES = 26
TURBINE_CAPACITY = 3.465     # MW per turbine

TARGET_COL = "Выработка. Результирующий расчет"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"
REPAIRS_COL = "Кол-во_ВЭУ_в_ремонте"

# Validation split: same season as test (Q1 2026)
VAL_START = "2025-01-01"
VAL_END = "2025-03-31 23:00:00"
