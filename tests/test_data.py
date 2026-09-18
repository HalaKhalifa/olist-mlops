"""Data tests: schema verification, range validation, null checks, and leakage firewall."""

from src.validation import data_validator


def test_valid_order_schema_passes(sample_order_dict):
    """Clean sample order must pass all Great Expectations data checks."""
    is_valid, errors = data_validator.validate(sample_order_dict)
    assert is_valid is True
    assert len(errors) == 0


def test_negative_financial_ranges_rejected(sample_order_dict):
    """Negative prices or freight values must be caught and rejected."""
    bad_order = sample_order_dict.copy()
    bad_order["total_price"] = -15.50
    bad_order["total_freight"] = -5.00

    is_valid, errors = data_validator.validate(bad_order)
    assert is_valid is False
    assert any("total_price" in e for e in errors)


def test_invalid_state_rejected(sample_order_dict):
    """Non-existent Brazilian state codes must be caught and rejected."""
    bad_order = sample_order_dict.copy()
    bad_order["customer_state"] = "INVALID_STATE"

    is_valid, errors = data_validator.validate(bad_order)
    assert is_valid is False
    assert any("customer_state" in e for e in errors)


def test_target_leakage_firewall(sample_order_dict):
    """Post-delivery columns must trigger strict target leakage violation."""
    leaky_order = sample_order_dict.copy()
    leaky_order["order_delivered_customer_date"] = "2017-10-31 21:47:42"

    is_valid, errors = data_validator.validate(leaky_order)
    assert is_valid is False
    assert any("Target leakage violation" in e for e in errors)


def test_is_late_leakage_rejected(sample_order_dict):
    """Passing ground truth target label is_late in inference input must be blocked."""
    leaky_order = sample_order_dict.copy()
    leaky_order["is_late"] = 1

    is_valid, errors = data_validator.validate(leaky_order)
    assert is_valid is False
    assert any("is_late" in e for e in errors)
