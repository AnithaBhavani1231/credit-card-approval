from __future__ import annotations

FORM_OPTIONS = {
    "gender": ["Female", "Male", "Other"],
    "own_car": ["No", "Yes"],
    "own_property": ["Yes", "No"],
    "income_type": [
        "Working",
        "Commercial associate",
        "State servant",
        "Pensioner",
        "Student",
    ],
    "education_level": [
        "Secondary",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ],
    "family_status": [
        "Married",
        "Single",
        "Civil marriage",
        "Separated",
        "Widow",
    ],
    "housing_type": [
        "House / apartment",
        "With parents",
        "Municipal apartment",
        "Rented apartment",
        "Office apartment",
        "Co-op apartment",
    ],
    "payment_status": [
        "Clean history",
        "Paid off this month",
        "1 month past due",
        "2 months past due",
        "3+ months past due",
    ],
}

DEFAULT_FORM_VALUES = {
    "gender": "Female",
    "own_car": "No",
    "own_property": "Yes",
    "children_count": "0",
    "annual_income": "65000",
    "income_type": "Working",
    "education_level": "Higher education",
    "family_status": "Married",
    "housing_type": "House / apartment",
    "employment_years": "5",
    "age": "34",
    "family_members": "2",
    "credit_inquiries_6m": "1",
    "existing_loan_balance": "8500",
    "payment_status": "Clean history",
    "mobile_phone": "1",
    "work_phone": "0",
    "phone": "1",
    "email": "1",
    "total_months": "24",
}

PAYMENT_STATUS_MAP = {
    "c": "Clean history",
    "x": "Clean history",
    "0": "Paid off this month",
    "1": "1 month past due",
    "2": "2 months past due",
    "3": "3+ months past due",
    "4": "3+ months past due",
    "5": "3+ months past due",
    "clean history": "Clean history",
    "paid off this month": "Paid off this month",
    "1 month past due": "1 month past due",
    "2 months past due": "2 months past due",
    "3+ months past due": "3+ months past due",
}

PAYMENT_STATUS_RISK = {
    "Clean history": 0,
    "Paid off this month": 0,
    "1 month past due": 1,
    "2 months past due": 1,
    "3+ months past due": 1,
}

PAYMENT_STATUS_OVERDUE_MONTHS = {
    "Clean history": 0,
    "Paid off this month": 0,
    "1 month past due": 1,
    "2 months past due": 2,
    "3+ months past due": 3,
}
