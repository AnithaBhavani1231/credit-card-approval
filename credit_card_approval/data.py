from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from credit_card_approval.config import (
    DEFAULT_SAMPLE_SIZE,
    RANDOM_STATE,
    SAMPLE_DATA_PATH,
    TARGET_COLUMN,
    ensure_project_directories,
)
from credit_card_approval.features import clean_application_data


COLUMN_ALIASES = {
    "code_gender": "gender",
    "flag_own_car": "own_car",
    "flag_own_realty": "own_property",
    "cnt_children": "children_count",
    "amt_income_total": "annual_income",
    "name_income_type": "income_type",
    "name_education_type": "education_level",
    "name_family_status": "family_status",
    "name_housing_type": "housing_type",
    "days_birth": "days_birth",
    "days_employed": "days_employed",
    "flag_mobil": "mobile_phone",
    "flag_work_phone": "work_phone",
    "flag_phone": "phone",
    "flag_email": "email",
    "cnt_fam_members": "family_members",
    "months_balance": "total_months",
    "status": "payment_status",
}


def load_credit_data(data_path: str | Path | None = None, sample_size: int = DEFAULT_SAMPLE_SIZE) -> pd.DataFrame:
    """Load a user dataset if available, otherwise create a realistic sample dataset."""
    ensure_project_directories()

    if data_path:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        frame = pd.read_csv(path)
    elif SAMPLE_DATA_PATH.exists():
        frame = pd.read_csv(SAMPLE_DATA_PATH)
    else:
        frame = generate_synthetic_applications(sample_size=sample_size)
        frame.to_csv(SAMPLE_DATA_PATH, index=False)

    frame = normalize_column_names(frame)
    return clean_application_data(frame)


def normalize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [
        column.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        for column in normalized.columns
    ]
    normalized = normalized.rename(columns=COLUMN_ALIASES)

    if "days_birth" in normalized and "age" not in normalized:
        normalized["age"] = (normalized["days_birth"].abs() / 365.25).round(1)

    if "days_employed" in normalized and "employment_years" not in normalized:
        years = normalized["days_employed"].abs() / 365.25
        years = years.mask(years > 80, 0)
        normalized["employment_years"] = years.round(1)

    return normalized


def generate_synthetic_applications(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Create a bank-like sample dataset for local training and demos.

    The generated data is intentionally realistic enough for development, but it is
    not a substitute for audited production banking data.
    """
    rng = np.random.default_rng(random_state)

    gender = rng.choice(["Female", "Male", "Other"], sample_size, p=[0.54, 0.44, 0.02])
    age = np.clip(rng.normal(39, 10.5, sample_size), 21, 68).round(0)
    children_count = np.clip(rng.poisson(0.7, sample_size), 0, 4)
    family_members = np.maximum(children_count + rng.choice([1, 2], sample_size, p=[0.34, 0.66]), 1)

    income_type = rng.choice(
        ["Working", "Commercial associate", "State servant", "Pensioner", "Student"],
        sample_size,
        p=[0.55, 0.21, 0.12, 0.09, 0.03],
    )
    education_level = rng.choice(
        ["Secondary", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
        sample_size,
        p=[0.48, 0.34, 0.1, 0.05, 0.03],
    )
    family_status = rng.choice(
        ["Married", "Single", "Civil marriage", "Separated", "Widow"],
        sample_size,
        p=[0.58, 0.22, 0.09, 0.07, 0.04],
    )
    housing_type = rng.choice(
        [
            "House / apartment",
            "With parents",
            "Municipal apartment",
            "Rented apartment",
            "Office apartment",
            "Co-op apartment",
        ],
        sample_size,
        p=[0.72, 0.1, 0.07, 0.06, 0.03, 0.02],
    )

    income_multiplier = np.select(
        [
            income_type == "Commercial associate",
            income_type == "State servant",
            income_type == "Pensioner",
            income_type == "Student",
        ],
        [1.2, 1.05, 0.62, 0.42],
        default=1.0,
    )
    education_multiplier = np.select(
        [
            education_level == "Higher education",
            education_level == "Academic degree",
            education_level == "Lower secondary",
        ],
        [1.24, 1.45, 0.78],
        default=1.0,
    )
    annual_income = rng.lognormal(mean=10.84, sigma=0.42, size=sample_size)
    annual_income = np.clip(annual_income * income_multiplier * education_multiplier, 15000, 260000).round(0)

    employment_base = np.clip((age - 21) * rng.uniform(0.15, 0.82, sample_size), 0, 38)
    employment_years = np.where(income_type == "Pensioner", 0, employment_base)
    employment_years = np.where(income_type == "Student", rng.uniform(0, 2, sample_size), employment_years)
    employment_years = employment_years.round(1)

    own_car = rng.choice(["Yes", "No"], sample_size, p=[0.38, 0.62])
    own_property = rng.choice(["Yes", "No"], sample_size, p=[0.68, 0.32])
    mobile_phone = np.ones(sample_size, dtype=int)
    work_phone = rng.binomial(1, 0.28, sample_size)
    phone = rng.binomial(1, 0.55, sample_size)
    email = rng.binomial(1, 0.42, sample_size)

    credit_inquiries_6m = np.clip(rng.poisson(1.1, sample_size), 0, 8)
    total_months = rng.integers(6, 61, sample_size)
    base_loan = rng.gamma(shape=2.1, scale=3900, size=sample_size)
    existing_loan_balance = np.clip(base_loan + annual_income * rng.uniform(0.02, 0.34, sample_size), 0, 130000)
    debt_to_income_ratio = existing_loan_balance / np.maximum(annual_income, 1)

    risk_score = (
        -1.25
        + 2.15 * debt_to_income_ratio
        + 0.22 * credit_inquiries_6m
        - 0.000006 * annual_income
        - 0.045 * employment_years
        + 0.22 * (income_type == "Student")
        + 0.18 * (housing_type == "Rented apartment")
        + 0.16 * (own_property == "No")
        + 0.12 * (children_count >= 3)
        + 0.18 * (age < 24)
    )
    reject_probability = 1 / (1 + np.exp(-risk_score))
    approved = (rng.random(sample_size) > reject_probability).astype(int)

    max_overdue_months = np.where(
        approved == 1,
        rng.choice([0, 1, 2], sample_size, p=[0.86, 0.12, 0.02]),
        rng.choice([0, 1, 2, 3, 4], sample_size, p=[0.32, 0.25, 0.2, 0.15, 0.08]),
    )
    payment_status = np.select(
        [
            max_overdue_months == 0,
            max_overdue_months == 1,
            max_overdue_months == 2,
            max_overdue_months >= 3,
        ],
        ["Clean history", "1 month past due", "2 months past due", "3+ months past due"],
        default="Paid off this month",
    )
    paid_off_mask = (max_overdue_months == 0) & (rng.random(sample_size) < 0.22)
    payment_status = np.where(paid_off_mask, "Paid off this month", payment_status)

    return pd.DataFrame(
        {
            "gender": gender,
            "own_car": own_car,
            "own_property": own_property,
            "children_count": children_count,
            "annual_income": annual_income,
            "income_type": income_type,
            "education_level": education_level,
            "family_status": family_status,
            "housing_type": housing_type,
            "employment_years": employment_years,
            "age": age,
            "mobile_phone": mobile_phone,
            "work_phone": work_phone,
            "phone": phone,
            "email": email,
            "family_members": family_members,
            "total_months": total_months,
            "max_overdue_months": max_overdue_months,
            "credit_inquiries_6m": credit_inquiries_6m,
            "existing_loan_balance": existing_loan_balance.round(0),
            "debt_to_income_ratio": debt_to_income_ratio.round(4),
            "payment_status": payment_status,
            TARGET_COLUMN: approved,
        }
    )
