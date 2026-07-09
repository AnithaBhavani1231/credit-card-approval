from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_card_approval.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN
from credit_card_approval.schemas import (
    DEFAULT_FORM_VALUES,
    PAYMENT_STATUS_MAP,
    PAYMENT_STATUS_OVERDUE_MONTHS,
    PAYMENT_STATUS_RISK,
)


def clean_application_data(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    cleaned["payment_status"] = cleaned.get("payment_status", DEFAULT_FORM_VALUES["payment_status"]).map(
        normalize_payment_status
    )
    cleaned["max_overdue_months"] = cleaned.get("max_overdue_months", cleaned["payment_status"].map(PAYMENT_STATUS_OVERDUE_MONTHS))
    cleaned["high_risk_history"] = cleaned["payment_status"].map(PAYMENT_STATUS_RISK).fillna(0).astype(int)

    for column, default_value in DEFAULT_FORM_VALUES.items():
        if column not in cleaned:
            cleaned[column] = default_value

    for column in NUMERIC_FEATURES:
        if column not in cleaned:
            cleaned[column] = 0
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["annual_income"] = cleaned["annual_income"].clip(lower=1)
    cleaned["existing_loan_balance"] = cleaned["existing_loan_balance"].clip(lower=0)
    cleaned["debt_to_income_ratio"] = (
        cleaned["existing_loan_balance"] / cleaned["annual_income"].replace(0, pd.NA)
    ).fillna(0)
    cleaned["debt_to_income_ratio"] = cleaned["debt_to_income_ratio"].clip(lower=0, upper=5)

    for column in CATEGORICAL_FEATURES:
        if column not in cleaned:
            cleaned[column] = DEFAULT_FORM_VALUES.get(column, "Unknown")
        cleaned[column] = cleaned[column].fillna(DEFAULT_FORM_VALUES.get(column, "Unknown")).astype(str).str.strip()

    cleaned["own_car"] = cleaned["own_car"].map(normalize_yes_no)
    cleaned["own_property"] = cleaned["own_property"].map(normalize_yes_no)
    cleaned["gender"] = cleaned["gender"].map(normalize_gender)

    if TARGET_COLUMN not in cleaned:
        cleaned[TARGET_COLUMN] = (cleaned["high_risk_history"] == 0).astype(int)

    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].clip(0, 1)

    return cleaned[FEATURE_COLUMNS + [TARGET_COLUMN]]


def normalize_payment_status(value: object) -> str:
    if pd.isna(value):
        return DEFAULT_FORM_VALUES["payment_status"]
    return PAYMENT_STATUS_MAP.get(str(value).strip().lower(), str(value).strip())


def normalize_yes_no(value: object) -> str:
    if pd.isna(value):
        return "No"
    text = str(value).strip().lower()
    if text in {"y", "yes", "1", "true"}:
        return "Yes"
    if text in {"n", "no", "0", "false"}:
        return "No"
    return str(value).strip().title()


def normalize_gender(value: object) -> str:
    if pd.isna(value):
        return "Other"
    text = str(value).strip().lower()
    if text in {"f", "female"}:
        return "Female"
    if text in {"m", "male"}:
        return "Male"
    return "Other"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
