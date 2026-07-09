from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SAMPLE_DATA_PATH = PROCESSED_DATA_DIR / "sample_credit_card_applications.csv"
MODEL_PATH = MODEL_DIR / "credit_card_approval_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

RANDOM_STATE = 42
DEFAULT_SAMPLE_SIZE = 3200

TARGET_COLUMN = "approved"

NUMERIC_FEATURES = [
    "children_count",
    "annual_income",
    "employment_years",
    "age",
    "mobile_phone",
    "work_phone",
    "phone",
    "email",
    "family_members",
    "total_months",
    "max_overdue_months",
    "credit_inquiries_6m",
    "existing_loan_balance",
    "debt_to_income_ratio",
    "high_risk_history",
]

CATEGORICAL_FEATURES = [
    "gender",
    "own_car",
    "own_property",
    "income_type",
    "education_level",
    "family_status",
    "housing_type",
    "payment_status",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def ensure_project_directories() -> None:
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
