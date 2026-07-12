"""Train and evaluate the customer churn model from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.churn_pipeline import (
    DEFAULT_DATA_PATH,
    RANDOM_STATE,
    build_pipeline,
    coefficient_table,
    load_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the telco churn classifier.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts")
    return parser.parse_args()


def save_churn_distribution(cleaned: pd.DataFrame, destination: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.2, 4.8))
    order = ["No", "Yes"]
    ax = sns.countplot(data=cleaned, x="Churn", order=order, hue="Churn", palette="Blues", legend=False)
    total = len(cleaned)
    for container in ax.containers:
        labels = [f"{int(bar.get_height()):,}\n({bar.get_height() / total:.1%})" for bar in container]
        ax.bar_label(container, labels=labels, padding=5)
    ax.set(title="Customer Churn Distribution", xlabel="Churned", ylabel="Customers")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    plt.tight_layout()
    plt.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, destination: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6.2, 5.0))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Stayed", "Churned"],
        yticklabels=["Stayed", "Churned"],
    )
    plt.title("Confusion Matrix — Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close()


def save_roc_curve(y_true: pd.Series, probabilities: pd.Series, destination: Path) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities)
    plt.figure(figsize=(6.5, 5.2))
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2.5, label=f"Logistic regression (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    plt.title("Receiver Operating Characteristic")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close()


def save_top_coefficients(model, destination: Path) -> None:
    top = coefficient_table(model).head(14).sort_values("coefficient")
    colors = ["#C44E52" if value < 0 else "#4C72B0" for value in top["coefficient"]]
    plt.figure(figsize=(9.2, 6.4))
    plt.barh(top["feature"], top["coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Strongest Logistic-Regression Signals")
    plt.xlabel("Coefficient (positive values indicate higher churn risk)")
    plt.tight_layout()
    plt.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close()


def train_and_evaluate(data_path: Path, reports_dir: Path, artifacts_dir: Path) -> dict[str, float]:
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    features, target, cleaned = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(
        build_pipeline(),
        X_train,
        y_train,
        scoring="roc_auc",
        cv=cross_validation,
        n_jobs=None,
    )

    metrics = {
        "rows": int(len(cleaned)),
        "features_before_encoding": int(features.shape[1]),
        "test_rows": int(len(X_test)),
        "churn_rate": float(target.mean()),
        "baseline_accuracy": float(accuracy_score(y_test, baseline_predictions)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "cv_roc_auc_mean": float(cv_auc.mean()),
        "cv_roc_auc_std": float(cv_auc.std()),
    }

    with (reports_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    joblib.dump(model, artifacts_dir / "churn_pipeline.joblib")

    save_churn_distribution(cleaned, figures_dir / "churn_distribution.png")
    save_confusion_matrix(y_test, predictions, figures_dir / "confusion_matrix.png")
    save_roc_curve(y_test, probabilities, figures_dir / "roc_curve.png")
    save_top_coefficients(model, figures_dir / "top_coefficients.png")

    return metrics


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(args.data, args.reports, args.artifacts)
    print("Customer churn pipeline completed successfully.\n")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name:>28}: {value:.4f}")
        else:
            print(f"{name:>28}: {value}")


if __name__ == "__main__":
    main()
