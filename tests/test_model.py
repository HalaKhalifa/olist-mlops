import numpy as np

from src.predict import prediction_service


def test_model_loaded_and_callable():
    """Verify that model is loaded and exposes predict and predict_proba."""
    model = prediction_service.model
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_prediction_output_contract(sample_order_dict):
    """Verify single prediction output structure and bounds."""
    result = prediction_service.predict_single(sample_order_dict)

    assert "prediction" in result
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("late", "on_time")
    assert 0.0 <= result["late_probability"] <= 1.0
    assert result["model_version"] == "1"
    assert result["latency_ms"] > 0.0


def test_reproducibility_parity_with_notebook(sample_order_dict):
    """Show that the pipeline output matches notebook output on the exact same input."""
    # From Notebook 6 and Notebook 5 evaluation:
    # The shipped frozen artifacts produce class 0 (on-time) and probability 0.4595.
    result = prediction_service.predict_single(sample_order_dict)

    assert result["prediction"] == 0
    assert result["label"] == "on_time"
    assert np.isclose(result["late_probability"], 0.4595, atol=1e-4)


def test_batch_prediction_shape(sample_batch_dict):
    """Verify batch prediction returns list of results matching input length."""
    orders = sample_batch_dict["orders"]
    results = prediction_service.predict_batch(orders)

    assert len(results) == len(orders)
    for res in results:
        assert res["prediction"] in (0, 1)
        assert 0.0 <= res["late_probability"] <= 1.0
