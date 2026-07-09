from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from credit_card_approval.config import FIGURES_DIR, TARGET_COLUMN, ensure_project_directories
from credit_card_approval.data import load_credit_data


def build_eda_charts(data_path: str | Path | None = None) -> list[Path]:
    ensure_project_directories()
    frame = load_credit_data(data_path=data_path)
    sns.set_theme(style="whitegrid", palette="Set2")

    output_paths: list[Path] = []

    output_paths.append(_save_count_plot(frame))
    output_paths.append(_save_income_distribution(frame))
    output_paths.append(_save_approval_by_income_type(frame))
    output_paths.append(_save_correlation_heatmap(frame))

    return output_paths


def _save_count_plot(frame: object) -> Path:
    path = FIGURES_DIR / "approval_count_plot.png"
    plt.figure(figsize=(7, 4))
    ax = sns.countplot(data=frame, x=TARGET_COLUMN)
    ax.set_title("Approval Decision Count")
    ax.set_xlabel("Decision")
    ax.set_ylabel("Applications")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Rejected", "Approved"])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _save_income_distribution(frame: object) -> Path:
    path = FIGURES_DIR / "annual_income_distribution.png"
    plt.figure(figsize=(8, 4.6))
    ax = sns.histplot(data=frame, x="annual_income", hue=TARGET_COLUMN, bins=35, kde=True)
    ax.set_title("Annual Income Distribution")
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Applicants")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _save_approval_by_income_type(frame: object) -> Path:
    path = FIGURES_DIR / "approval_by_income_type.png"
    plt.figure(figsize=(9, 4.8))
    grouped = frame.groupby("income_type")[TARGET_COLUMN].mean().sort_values(ascending=False)
    ax = sns.barplot(x=grouped.index, y=grouped.values)
    ax.set_title("Approval Rate by Income Type")
    ax.set_xlabel("Income Type")
    ax.set_ylabel("Approval Rate")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _save_correlation_heatmap(frame: object) -> Path:
    path = FIGURES_DIR / "numeric_feature_correlation.png"
    numeric_frame = frame.select_dtypes(include=["number"])
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(numeric_frame.corr(), cmap="vlag", center=0, linewidths=0.4)
    ax.set_title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
