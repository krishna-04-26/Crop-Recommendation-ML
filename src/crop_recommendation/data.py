"""Dataset loading, validation, and stratified splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class DataSplits:
    """Feature and label frames for a reproducible 60/20/20 split."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the assignment dataset before any modelling."""
    frame = pd.read_csv(path)
    expected = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    actual = set(frame.columns)
    if actual != expected:
        raise ValueError(
            f"Unexpected columns. Expected {sorted(expected)}, got {sorted(actual)}."
        )
    frame = frame.loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    if frame.empty:
        raise ValueError("Dataset contains no rows.")
    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values; resolve them before training.")
    if frame.duplicated().any():
        raise ValueError("Dataset contains exact duplicate rows; resolve them before training.")
    for feature in FEATURE_COLUMNS:
        frame[feature] = pd.to_numeric(frame[feature], errors="raise")
    if (frame[TARGET_COLUMN].astype(str).str.strip() == "").any():
        raise ValueError("Dataset contains an empty crop label.")
    if frame[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")
    return frame


def quality_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Return report-ready, JSON-serialisable data-quality details."""
    return {
        "rows": int(len(frame)),
        "features": list(FEATURE_COLUMNS),
        "class_count": int(frame[TARGET_COLUMN].nunique()),
        "missing_values": {key: int(value) for key, value in frame.isna().sum().items()},
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "class_distribution": {
            str(key): int(value)
            for key, value in frame[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "feature_ranges": {
            feature: {"min": float(frame[feature].min()), "max": float(frame[feature].max())}
            for feature in FEATURE_COLUMNS
        },
    }


def make_splits(frame: pd.DataFrame, random_state: int = 42) -> DataSplits:
    """Create the required stratified 60/20/20 train/validation/test split."""
    X = frame.loc[:, list(FEATURE_COLUMNS)]
    y = frame[TARGET_COLUMN]
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, test_size=0.40, random_state=random_state, stratify=y
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.50,
        random_state=random_state,
        stratify=y_remaining,
    )
    return DataSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )
