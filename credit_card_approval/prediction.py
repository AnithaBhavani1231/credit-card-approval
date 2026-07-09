from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from credit_card_approval.config import FEATURE_COLUMNS, MODEL_PATH
from credit_card_approval.features import clean_application_data, normalize_payment_status
from credit_card_approval.schemas import DEFAULT_FORM_VALUES, PAYMENT_STATUS_OVERDUE_MONTHS, PAYMENT_STATUS_RISK

MINIMUM_ANNUAL_INCOME = 12000
AUTO_REJECT_DEBT_TO_INCOME = 1.0
AUTO_REJECT_CREDIT_INQUIRIES = 8
SERIOUS_OVERDUE_STATUS = "3+ months past due"


def load_model(model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict_application(raw_payload: dict[str, Any], model_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = normalize_prediction_payload(raw_payload)
    frame = clean_application_data(pd.DataFrame([payload]))
    model_payload = model_payload or load_model()
    model = model_payload["model"]

    prediction = int(model.predict(frame[FEATURE_COLUMNS])[0])
    approval_probability = get_approval_probability(model, frame[FEATURE_COLUMNS], prediction)
    policy_rejections = evaluate_policy_rejections(payload)

    if policy_rejections:
        decision = "Rejected"
        approval_probability = min(approval_probability, 0.24)
        risk_level = "High risk"
    else:
        decision = "Approved" if prediction == 1 else "Rejected"
        risk_level = classify_risk(approval_probability)

    return {
        "decision": decision,
        "approval_probability": round(float(approval_probability), 4),
        "approval_probability_percent": round(float(approval_probability) * 100, 1),
        "risk_level": risk_level,
        "model_name": model_payload.get("best_model_name", "Model"),
        "reasons": build_reason_codes(payload, approval_probability, policy_rejections),
    }


def normalize_prediction_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(DEFAULT_FORM_VALUES)
    payload.update({key: value for key, value in raw_payload.items() if value not in [None, ""]})

    numeric_fields = [
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
        "credit_inquiries_6m",
        "existing_loan_balance",
    ]

    for field in numeric_fields:
        try:
            payload[field] = float(payload[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field.replace('_', ' ').title()} must be a valid number.") from exc

    if payload["annual_income"] <= 0:
        raise ValueError("Annual income must be greater than zero.")
    if payload["age"] < 18:
        raise ValueError("Applicant age must be at least 18.")
    if payload["existing_loan_balance"] < 0:
        raise ValueError("Existing loan balance cannot be negative.")

    payload["payment_status"] = normalize_payment_status(payload.get("payment_status"))
    payload["max_overdue_months"] = PAYMENT_STATUS_OVERDUE_MONTHS.get(payload["payment_status"], 0)
    payload["high_risk_history"] = PAYMENT_STATUS_RISK.get(payload["payment_status"], 0)
    payload["debt_to_income_ratio"] = payload["existing_loan_balance"] / payload["annual_income"]

    return payload


def get_approval_probability(model: Any, frame: pd.DataFrame, prediction: int) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        return float(probabilities[1])
    return 0.82 if prediction == 1 else 0.18


def classify_risk(approval_probability: float) -> str:
    if approval_probability >= 0.72:
        return "Low risk"
    if approval_probability >= 0.46:
        return "Needs review"
    return "High risk"


def evaluate_policy_rejections(payload: dict[str, Any]) -> list[str]:
    rejections: list[str] = []

    if payload["annual_income"] < MINIMUM_ANNUAL_INCOME:
        rejections.append("Annual income is below the minimum eligibility level for a new credit card.")
    if payload["debt_to_income_ratio"] >= AUTO_REJECT_DEBT_TO_INCOME:
        rejections.append("Existing loan balance is too high compared with annual income.")
    if payload["payment_status"] == SERIOUS_OVERDUE_STATUS:
        rejections.append("Payment history shows serious overdue records.")
    if payload["credit_inquiries_6m"] >= AUTO_REJECT_CREDIT_INQUIRIES:
        rejections.append("Recent credit inquiry count is too high for automatic approval.")

    return rejections


def build_reason_codes(
    payload: dict[str, Any],
    approval_probability: float,
    policy_rejections: list[str] | None = None,
) -> list[str]:
    reasons: list[str] = list(policy_rejections or [])

    if payload["debt_to_income_ratio"] >= 0.45 and not any("loan balance" in reason for reason in reasons):
        reasons.append("Debt level is high compared with annual income.")
    if payload["credit_inquiries_6m"] >= 4 and not any("credit inquiry" in reason for reason in reasons):
        reasons.append("Recent credit inquiry count is elevated.")
    if payload["payment_status"] in {"1 month past due", "2 months past due", "3+ months past due"} and not any(
        "overdue" in reason for reason in reasons
    ):
        reasons.append("Credit history includes recent overdue payments.")
    if payload["employment_years"] < 1 and payload["income_type"] not in {"Pensioner", "Student"}:
        reasons.append("Employment history is short for the selected income type.")
    if payload["annual_income"] < 25000 and not any("Annual income" in reason for reason in reasons):
        reasons.append("Annual income is below the preferred approval range.")
    if payload["age"] < 23:
        reasons.append("Applicant profile may have limited credit history.")

    if not reasons:
        if approval_probability >= 0.72:
            reasons.append("Profile shows stable income, manageable debt, and clean repayment history.")
        elif approval_probability >= 0.46:
            reasons.append("Profile is close to the approval boundary and should be reviewed by an analyst.")
        else:
            reasons.append("Model identified a combined risk pattern across the submitted profile.")

    return reasons
