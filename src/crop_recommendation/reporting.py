"""Generate a concise, evidence-based report for the assignment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import FEATURE_COLUMNS


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_assignment_report(
    path: str | Path,
    quality: dict[str, object],
    comparison: pd.DataFrame,
    selected_configuration: dict[str, object],
    test_metrics: dict[str, float],
    feature_importance: pd.DataFrame,
    errors: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    """Write a report containing all common and topic-specific deliverables."""
    selected = comparison.loc[
        (comparison["criterion"] == selected_configuration["criterion"])
        & (comparison["max_depth"].fillna(-1) == (selected_configuration["max_depth"] if selected_configuration["max_depth"] is not None else -1))
    ].iloc[0]
    sensitive = (
        sensitivity.groupby("feature_changed")["prediction_changed"].sum().sort_values(ascending=False)
    )
    sensitive_text = ", ".join(f"{feature}: {int(count)} changes" for feature, count in sensitive.items())
    importance_text = ", ".join(
        f"{row.feature} ({row.importance:.3f})" for row in feature_importance.itertuples(index=False)
    )
    comparison_rows = "\n".join(
        f"| {row.configuration} | {_percent(row.accuracy)} | {_percent(row.precision_macro)} | {_percent(row.recall_macro)} | {_percent(row.f1_macro)} |"
        for row in comparison.itertuples(index=False)
    )
    top_error_pairs = (
        errors.groupby(["actual_label", "predicted_label"]).size().sort_values(ascending=False).head(5)
    )
    error_pairs_text = "; ".join(
        f"{actual} -> {predicted} ({count})" for (actual, predicted), count in top_error_pairs.items()
    ) or "No final-test errors."

    content = f"""# Crop Recommendation under Changing Soil Conditions

## 1. Problem formulation

This is a multiclass classification problem: recommend one crop label from measured soil nutrients and weather conditions. The input features are {", ".join(FEATURE_COLUMNS)}; the target is `label`. Classification is appropriate because the output is one of a fixed set of crop classes. It can support preliminary crop-suitability screening, but is not a substitute for local agronomic advice.

## 2. Dataset analysis

The supplied dataset has {quality["rows"]} samples, {quality["class_count"]} crop classes, and seven numerical features. It has no missing values and {quality["exact_duplicate_rows"]} exact duplicate rows. Classes are balanced according to `data_quality.json`. Feature ranges are stored there for transparent input validation. The dataset reference is the original supplied repository; its primary collection provenance and licence should be verified before reuse.

## 3. Preprocessing and justification

All supplied features are numerical, so no encoding is needed. No missing-value treatment is applied because validation finds none. Decision Trees split raw values and therefore do not require scaling, normalization, or text transformation. All seven supplied agronomic features are retained. The data are split with stratification and a fixed random seed into 60% training, 20% validation, and 20% test data.

## 4. Model development and mathematical principle

An entropy-based Decision Tree is used, as required. At each node it selects the split with maximum information gain:

`Entropy(S) = -Σ p(c) log2 p(c)`

`IG(S, A) = Entropy(S) - Σ (|S_v| / |S|) Entropy(S_v)`

The candidate configurations differ only in maximum depth: 5, 10, and unlimited. The random seed is fixed at 42.

## 5. Validation-based model selection

The test set is not used for model selection. Macro F1 is the primary selection criterion because it weights every crop class equally; accuracy is secondary. For an exact metric tie, the shallower tree is chosen for simpler, more interpretable rules.

| Configuration | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
{comparison_rows}

The selected configuration is `{selected_configuration}` with validation macro F1 {_percent(float(selected.f1_macro))}.

## 6. Final test evaluation

The selected model is evaluated once on the held-out test set. Accuracy: {_percent(test_metrics["accuracy"])}; macro precision: {_percent(test_metrics["precision_macro"])}; macro recall: {_percent(test_metrics["recall_macro"])}; macro F1: {_percent(test_metrics["f1_macro"])}; macro one-vs-rest ROC-AUC: {_percent(test_metrics["roc_auc_macro_ovr"])}. See `confusion_matrix.png` and `per_class_metrics.csv` for class-level results.

## 7. Assignment-specific sensitivity and rule investigation

Three representative test samples (positions 0, 50, and 100) are used. One feature at a time is increased and decreased by 20%, keeping the remaining inputs fixed. The full original/modified value and prediction table is in `sensitivity_analysis.csv`. Prediction changes by feature are: {sensitive_text}. This identifies features to which the selected model is locally sensitive for these samples. `decision_rules.txt` contains all learned rules; perturbed predictions in the sensitivity table are the rule-robustness test.

## 8. Visualisation and error analysis

`class_distribution.png` documents class balance. `feature_importance.png` shows the tree's impurity-based feature importance: {importance_text}. `confusion_matrix.png` displays final-test errors. There are {len(errors)} final-test misclassifications; the most frequent actual-to-predicted pairs are: {error_pairs_text}. Representative records are provided in `misclassified_samples.csv`.

## 9. Conclusion, limitations, and recommendation

The selected Decision Tree performs strongly on this held-out dataset and is transparent enough to explain using rules. However, its recommendations should not be deployed directly: the data do not establish geographic or seasonal coverage, and key farm factors such as soil texture, irrigation, pests, market conditions, and local expert constraints are absent. Before real-world use, validate against local, recent data and involve agronomists. The command-line demo rejects measurements outside observed data ranges to avoid unsupported extrapolation.
"""
    Path(path).write_text(content, encoding="utf-8")
