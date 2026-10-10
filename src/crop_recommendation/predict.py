"""Command-line prediction demo for the saved crop Decision Tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from . import FEATURE_COLUMNS


def predict(bundle_path: str | Path, values: dict[str, float]) -> dict[str, object]:
    """Validate one observation and return a prediction with top probabilities."""
    bundle = joblib.load(bundle_path)
    ranges = bundle["feature_ranges"]
    for feature in FEATURE_COLUMNS:
        value = values[feature]
        minimum, maximum = ranges[feature]["min"], ranges[feature]["max"]
        if not minimum <= value <= maximum:
            raise ValueError(
                f"{feature}={value} is outside the observed training range [{minimum}, {maximum}]."
            )
    frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    model = bundle["model"]
    probabilities = model.predict_proba(frame)[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda item: item[1], reverse=True)
    return {
        "prediction": ranked[0][0],
        "confidence": round(float(ranked[0][1]), 4),
        "top_3_probabilities": [
            {"label": label, "probability": round(float(probability), 4)}
            for label, probability in ranked[:3]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a crop from soil and weather measurements.")
    parser.add_argument("--model", default="artifacts/decision_tree_model.joblib")
    for feature in FEATURE_COLUMNS:
        parser.add_argument(f"--{feature}", type=float, required=True)
    args = parser.parse_args()
    values = {feature: getattr(args, feature) for feature in FEATURE_COLUMNS}
    print(json.dumps(predict(args.model, values), indent=2))


if __name__ == "__main__":
    main()
