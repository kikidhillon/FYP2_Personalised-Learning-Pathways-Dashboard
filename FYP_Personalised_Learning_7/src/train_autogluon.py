"""Optional AutoGluon comparison using the same student-grouped folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.data_access import CENTRAL_WORKBOOK, load_ml_semester
from src.train_risk_model import FEATURES, GROUP, TARGET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=CENTRAL_WORKBOOK)
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path("models/autogluon"))
    args = parser.parse_args()

    df = load_ml_semester(args.data)
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(int)
    features = [column for column in FEATURES if column in df.columns]
    X, y, groups = df[features], df[TARGET], df[GROUP]

    positive_groups = df.loc[y.eq(1), GROUP].nunique()
    negative_groups = df.loc[y.eq(0), GROUP].nunique()
    n_splits = min(3, positive_groups, negative_groups)
    if n_splits < 2:
        raise ValueError("Not enough positive and negative student groups for evaluation.")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        train_data = X.iloc[train_index].copy()
        train_data[TARGET] = y.iloc[train_index].to_numpy()
        test_X = X.iloc[test_index].copy()
        test_y = y.iloc[test_index]

        predictor = TabularPredictor(
            label=TARGET,
            problem_type="binary",
            eval_metric="average_precision",
            path=str(args.output / f"fold_{fold}"),
        ).fit(
            train_data=train_data,
            presets="medium_quality",
            time_limit=args.time_limit,
        )

        probability = predictor.predict_proba(test_X).iloc[:, -1].to_numpy()
        prediction = (probability >= 0.5).astype(int)
        metrics.append(
            {
                "fold": fold,
                "pr_auc": average_precision_score(test_y, probability),
                "recall": recall_score(test_y, prediction, zero_division=0),
                "precision": precision_score(test_y, prediction, zero_division=0),
                "f1": f1_score(test_y, prediction, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(test_y, prediction),
            }
        )

    results = pd.DataFrame(metrics)
    print(results)
    print("\nMean metrics\n", results.mean(numeric_only=True))
    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/autogluon_fold_metrics.csv", index=False)


if __name__ == "__main__":
    main()
