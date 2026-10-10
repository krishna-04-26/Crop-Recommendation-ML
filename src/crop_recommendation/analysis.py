"""Assignment-specific sensitivity, rules, error, and plot helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

from . import FEATURE_COLUMNS
from .data import DataSplits


def save_class_distribution(frame: pd.DataFrame, path: str | Path) -> None:
    """Save the balanced target-class distribution before splitting."""
    counts = frame["label"].value_counts().sort_index().reset_index()
    counts.columns = ["label", "count"]
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(data=counts, x="label", y="count", hue="label", palette="crest", legend=False, ax=axis)
    axis.set(title="Crop Class Distribution", xlabel="Crop label", ylabel="Samples")
    axis.tick_params(axis="x", rotation=55)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def sensitivity_analysis(
    model: DecisionTreeClassifier, splits: DataSplits, sample_positions: tuple[int, ...] = (0, 50, 100)
) -> pd.DataFrame:
    """Perturb one feature at a time by +/-20% for three test samples."""
    rows: list[dict[str, object]] = []
    for position in sample_positions:
        sample = splits.X_test.iloc[position].copy()
        actual = splits.y_test.iloc[position]
        baseline = model.predict(pd.DataFrame([sample]))[0]
        for feature in FEATURE_COLUMNS:
            for direction, multiplier in (("increase_20_percent", 1.2), ("decrease_20_percent", 0.8)):
                perturbed = sample.copy()
                perturbed[feature] = float(sample[feature]) * multiplier
                prediction = model.predict(pd.DataFrame([perturbed]))[0]
                rows.append(
                    {
                        "test_position": position,
                        "actual_label": actual,
                        "baseline_prediction": baseline,
                        "feature_changed": feature,
                        "perturbation": direction,
                        "original_value": float(sample[feature]),
                        "modified_value": float(perturbed[feature]),
                        "modified_prediction": prediction,
                        "prediction_changed": bool(prediction != baseline),
                    }
                )
    return pd.DataFrame(rows)


def error_analysis(model: DecisionTreeClassifier, splits: DataSplits) -> pd.DataFrame:
    """List representative false classifications for discussion in the report."""
    predictions = model.predict(splits.X_test)
    result = splits.X_test.copy()
    result["actual_label"] = splits.y_test.to_numpy()
    result["predicted_label"] = predictions
    result["correct"] = result["actual_label"] == result["predicted_label"]
    return result.loc[~result["correct"]].reset_index(names="source_index")


def write_rules(model: DecisionTreeClassifier, path: str | Path) -> None:
    """Write all learned, readable Decision Tree split rules."""
    Path(path).write_text(
        export_text(model, feature_names=list(FEATURE_COLUMNS)), encoding="utf-8"
    )


def save_confusion_matrix(model: DecisionTreeClassifier, splits: DataSplits, path: str | Path) -> None:
    """Save a labelled final-test confusion matrix."""
    predictions = model.predict(splits.X_test)
    matrix = confusion_matrix(splits.y_test, predictions, labels=model.classes_)
    figure, axis = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        ax=axis,
    )
    axis.set(xlabel="Predicted crop", ylabel="Actual crop", title="Final Test Confusion Matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_feature_importance(model: DecisionTreeClassifier, path: str | Path) -> pd.DataFrame:
    """Save and return impurity-based feature importance."""
    importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(data=importance, x="importance", y="feature", hue="feature", palette="viridis", legend=False, ax=axis)
    axis.set(title="Decision Tree Feature Importance", xlabel="Importance", ylabel="Feature")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return importance
