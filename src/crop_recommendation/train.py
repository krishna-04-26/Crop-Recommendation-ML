"""Run the complete, reproducible assignment experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .analysis import (
    error_analysis,
    save_class_distribution,
    save_confusion_matrix,
    save_feature_importance,
    sensitivity_analysis,
    write_rules,
)
from .data import load_dataset, make_splits, quality_summary
from .modeling import evaluate_final_model, select_model
from .reporting import write_assignment_report


def _write_splits(output_directory: Path, splits) -> None:
    for name, frame in (
        ("X_train", splits.X_train),
        ("X_validation", splits.X_validation),
        ("X_test", splits.X_test),
        ("y_train", splits.y_train),
        ("y_validation", splits.y_validation),
        ("y_test", splits.y_test),
    ):
        frame.to_csv(output_directory / f"{name}.csv", index=False)


def run(data_path: str | Path, artifact_directory: str | Path, report_directory: str | Path) -> dict[str, object]:
    """Run all stages and return the final evaluation summary."""
    artifacts = Path(artifact_directory)
    reports = Path(report_directory)
    processed = Path("data/processed")
    for directory in (artifacts, reports, processed):
        directory.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(data_path)
    quality = quality_summary(frame)
    splits = make_splits(frame)
    _write_splits(processed, splits)
    selected = select_model(splits)
    test_metrics, per_class = evaluate_final_model(selected.model, splits)

    selected.comparison.to_csv(reports / "model_comparison.csv", index=False)
    per_class.to_csv(reports / "per_class_metrics.csv", index=False)
    (reports / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    (reports / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    save_class_distribution(frame, reports / "class_distribution.png")
    save_confusion_matrix(selected.model, splits, reports / "confusion_matrix.png")
    importance = save_feature_importance(selected.model, reports / "feature_importance.png")
    importance.to_csv(reports / "feature_importance.csv", index=False)
    sensitivity = sensitivity_analysis(selected.model, splits)
    sensitivity.to_csv(reports / "sensitivity_analysis.csv", index=False)
    errors = error_analysis(selected.model, splits)
    errors.to_csv(reports / "misclassified_samples.csv", index=False)
    write_rules(selected.model, reports / "decision_rules.txt")
    write_assignment_report(
        reports / "assignment_report.md",
        quality,
        selected.comparison,
        selected.configuration,
        test_metrics,
        importance,
        errors,
        sensitivity,
    )
    joblib.dump(
        {
            "model": selected.model,
            "configuration": selected.configuration,
            "feature_ranges": quality["feature_ranges"],
            "feature_names": quality["features"],
        },
        artifacts / "decision_tree_model.joblib",
    )
    return {"configuration": selected.configuration, **test_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the crop Decision Tree.")
    parser.add_argument("--data", default="Crop_recommendation.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--reports", default="reports/generated")
    args = parser.parse_args()
    summary = run(args.data, args.artifacts, args.reports)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
