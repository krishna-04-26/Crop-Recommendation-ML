from crop_recommendation import FEATURE_COLUMNS
from crop_recommendation.data import load_dataset, make_splits, quality_summary


def test_dataset_has_expected_quality():
    frame = load_dataset("Crop_recommendation.csv")
    summary = quality_summary(frame)
    assert summary["rows"] == 2200
    assert summary["class_count"] == 22
    assert summary["exact_duplicate_rows"] == 0
    assert all(value == 0 for value in summary["missing_values"].values())
    assert list(frame.columns) == [*FEATURE_COLUMNS, "label"]


def test_stratified_split_sizes_and_classes():
    splits = make_splits(load_dataset("Crop_recommendation.csv"))
    assert len(splits.X_train) == 1320
    assert len(splits.X_validation) == len(splits.X_test) == 440
    assert splits.y_train.nunique() == splits.y_validation.nunique() == splits.y_test.nunique() == 22
