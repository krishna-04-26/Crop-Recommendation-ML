"""Decision Tree model selection and final evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

from .data import DataSplits


CANDIDATE_CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {"criterion": "entropy", "max_depth": 5},
    {"criterion": "entropy", "max_depth": 10},
    {"criterion": "entropy", "max_depth": None},
)


@dataclass(frozen=True)
class SelectionResult:
    model: DecisionTreeClassifier
    configuration: dict[str, Any]
    comparison: pd.DataFrame


def metric_row(y_true: pd.Series, predictions, probabilities=None, classes=None) -> dict[str, float]:
    """Compute macro multiclass metrics, including ROC-AUC when available."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
    if probabilities is not None and classes is not None:
        binary_targets = label_binarize(y_true, classes=classes)
        metrics["roc_auc_macro_ovr"] = float(
            roc_auc_score(binary_targets, probabilities, average="macro", multi_class="ovr")
        )
    return metrics


def select_model(splits: DataSplits, random_state: int = 42) -> SelectionResult:
    """Choose a model using validation macro F1, never the test set."""
    rows: list[dict[str, Any]] = []
    models: list[tuple[DecisionTreeClassifier, dict[str, Any]]] = []
    for config in CANDIDATE_CONFIGURATIONS:
        model = DecisionTreeClassifier(random_state=random_state, **config)
        model.fit(splits.X_train, splits.y_train)
        scores = metric_row(splits.y_validation, model.predict(splits.X_validation))
        rows.append({"configuration": str(config), **config, **scores})
        models.append((model, config.copy()))

    comparison = pd.DataFrame(rows)
    # Prefer best macro F1, then accuracy, then the simplest equally performing tree.
    ranked = comparison.assign(
        _complexity=comparison["max_depth"].fillna(float("inf"))
    ).sort_values(
        ["f1_macro", "accuracy", "_complexity"], ascending=[False, False, True], kind="stable"
    )
    selected_index = int(ranked.index[0])
    model, config = models[selected_index]
    return SelectionResult(model=model, configuration=config, comparison=comparison)


def evaluate_final_model(model: DecisionTreeClassifier, splits: DataSplits) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate the chosen model once on the untouched final test set."""
    predictions = model.predict(splits.X_test)
    probabilities = model.predict_proba(splits.X_test)
    metrics = metric_row(splits.y_test, predictions, probabilities, model.classes_)
    precision, recall, f1, support = precision_recall_fscore_support(
        splits.y_test, predictions, labels=model.classes_, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "label": model.classes_,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    return metrics, per_class
