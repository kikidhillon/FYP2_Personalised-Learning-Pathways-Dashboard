"""Train and evaluate the at-risk classifier with student-grouped folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_access import CENTRAL_WORKBOOK, load_ml_semester


TARGET = "next_at_risk_flag"
GROUP = "student_key"
FEATURES = [
    "programme",
    "semester_seq",
    "year",
    "period",
    "occurrence",
    "gpa",
    "cgpa",
    "standing",
    "semester_credit_hours",
    "accumulated_credit_hours",
    "module_count",
    "failed_module_count",
    "previous_gpa",
    "previous_cgpa",
    "previous_standing",
    "gpa_change",
    "cgpa_change",
]


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", classifier)])


def choose_threshold(y_true: pd.Series, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def calculate_metrics(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "pr_auc": average_precision_score(y_true, probability),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=CENTRAL_WORKBOOK,
        help="Central .xlsx workbook or prepared ml_semester.csv",
    )
    parser.add_argument("--model", type=Path, default=Path("models/risk_model.joblib"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()

    data = load_ml_semester(args.data)
    data = data[data[TARGET].notna()].copy()
    data[TARGET] = data[TARGET].astype(int)
    features = [column for column in FEATURES if column in data.columns]
    X, y, groups = data[features], data[TARGET], data[GROUP]

    positive_groups = data.loc[y.eq(1), GROUP].nunique()
    negative_groups = data.loc[y.eq(0), GROUP].nunique()
    n_splits = min(3, positive_groups, negative_groups)
    if data[GROUP].nunique() < 4 or y.nunique() < 2 or n_splits < 2:
        raise ValueError("Not enough positive and negative student groups for grouped evaluation.")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probability = np.full(len(data), np.nan)
    fold_indices = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        model = build_pipeline(X.iloc[train_index])
        model.fit(X.iloc[train_index], y.iloc[train_index])
        probability = model.predict_proba(X.iloc[test_index])[:, 1]
        oof_probability[test_index] = probability
        fold_indices.append((fold, test_index))

    threshold = choose_threshold(y, oof_probability)
    fold_metrics = []
    for fold, test_index in fold_indices:
        metrics = calculate_metrics(y.iloc[test_index], oof_probability[test_index], threshold)
        metrics.update(
            {
                "fold": fold,
                "test_students": groups.iloc[test_index].nunique(),
                "test_rows": len(test_index),
                "positive_rows": int(y.iloc[test_index].sum()),
            }
        )
        fold_metrics.append(metrics)

    overall = calculate_metrics(y, oof_probability, threshold)
    overall.update(
        {
            "fold": "overall_oof",
            "test_students": groups.nunique(),
            "test_rows": len(data),
            "positive_rows": int(y.sum()),
        }
    )

    final_model = build_pipeline(X)
    final_model.fit(X, y)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final_model,
            "threshold": threshold,
            "features": features,
            "target": TARGET,
            "training_rows": len(data),
            "training_students": groups.nunique(),
        },
        args.model,
    )

    pd.DataFrame(fold_metrics + [overall]).to_csv(
        args.reports / "risk_model_metrics.csv", index=False
    )
    predictions = data[[GROUP, "semester_seq", TARGET]].copy()
    predictions["risk_probability"] = oof_probability
    predictions["predicted_at_risk"] = (oof_probability >= threshold).astype(int)
    predictions.to_csv(args.reports / "risk_model_oof_predictions.csv", index=False)

    print(pd.Series(overall).to_string())
    print(f"\nSaved model to {args.model}")


if __name__ == "__main__":
    main()

