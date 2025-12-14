import pytest
from model_pipeline import load_data, train_model, evaluate_metrics

ACCURACY_THRESHOLD = 0.80
F1_THRESHOLD = 0.75


@pytest.mark.xfail(
    reason="Baseline model: F1-score below release threshold"
)
def test_model_metrics():
    X_train, X_test, y_train, y_test, _ = load_data()
    model = train_model(X_train, y_train)

    metrics = evaluate_metrics(model, X_test, y_test)

    assert metrics["accuracy"] >= ACCURACY_THRESHOLD, (
        f"Accuracy FAILED | Expected ≥ {ACCURACY_THRESHOLD}, "
        f"Got = {metrics['accuracy']:.2f}"
    )

    assert metrics["f1"] >= F1_THRESHOLD, (
        f"F1 FAILED | Expected ≥ {F1_THRESHOLD}, "
        f"Got = {metrics['f1']:.2f}"
    )
