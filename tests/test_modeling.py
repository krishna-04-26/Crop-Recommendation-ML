from crop_recommendation.data import load_dataset, make_splits
from crop_recommendation.modeling import evaluate_final_model, select_model


def test_selection_and_final_evaluation_produce_valid_metrics():
    splits = make_splits(load_dataset("Crop_recommendation.csv"))
    selected = select_model(splits)
    metrics, per_class = evaluate_final_model(selected.model, splits)
    assert len(selected.comparison) == 3
    assert selected.configuration["criterion"] == "entropy"
    assert set(metrics) == {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_macro_ovr",
    }
    assert all(0 <= score <= 1 for score in metrics.values())
    assert len(per_class) == 22
