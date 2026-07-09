from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from credit_card_approval.config import (
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    ensure_project_directories,
)
from credit_card_approval.data import load_credit_data
from credit_card_approval.features import build_preprocessor


def train_best_model(data_path: str | Path | None = None, sample_size: int | None = None) -> dict[str, Any]:
    ensure_project_directories()
    frame = load_credit_data(data_path=data_path, sample_size=sample_size or 3200)

    x = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.22,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    candidates = build_model_candidates()
    results: list[dict[str, Any]] = []
    trained_models: dict[str, Pipeline] = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        results.append(evaluate_classifier(name, y_test, predictions))
        trained_models[name] = pipeline

    ranked_results = sorted(results, key=lambda item: (item["f1_score"], item["accuracy"]), reverse=True)
    best_result = ranked_results[0]
    best_model = trained_models[best_result["model_name"]]

    payload = {
        "model": best_model,
        "best_model_name": best_result["model_name"],
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "best_model": best_result,
            "all_models": ranked_results,
            "training_rows": int(len(x_train)),
            "testing_rows": int(len(x_test)),
            "approval_rate": float(y.mean()),
        },
    }

    joblib.dump(payload, MODEL_PATH)
    write_metrics(payload["metrics"])
    return payload["metrics"]


def build_model_candidates() -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=18, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=260,
            max_depth=13,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = XGBClassifier(
            n_estimators=240,
            learning_rate=0.055,
            max_depth=4,
            subsample=0.86,
            colsample_bytree=0.88,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    except ImportError:
        candidates["XGBoost"] = RandomForestClassifier(
            n_estimators=180,
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    return candidates


def evaluate_classifier(model_name: str, y_true: pd.Series, y_pred: Any) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, target_names=["Rejected", "Approved"], output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred).tolist()

    return {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": matrix,
        "classification_report": report,
    }


def write_metrics(metrics: dict[str, Any]) -> None:
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

