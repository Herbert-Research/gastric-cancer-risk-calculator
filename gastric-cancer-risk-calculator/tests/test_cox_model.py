import pytest
from models.cox_model import CoxModel


@pytest.fixture
def mock_cox_config():
    """Returns a minimal valid configuration for the CoxModel."""
    return {
        "id": "test_model",
        "name": "Test Cox Model",
        "baseline_survival": {
            "5_year_estimate": 0.80,
            "10_year_estimate": 0.70
        },
        "variables": {
            "age": {
                "type": "continuous",
                "coefficient": 0.05
            },
            "sex": {
                "type": "categorical",
                "reference": "male",
                "categories": {
                    "male": 0.0,
                    "female": -0.5
                }
            }
        }
    }


def test_cox_model_initialization(mock_cox_config):
    model = CoxModel(mock_cox_config)
    assert model.name == "Test Cox Model"
    assert model.baseline_5yr == 0.80


def test_linear_predictor_calculation(mock_cox_config):
    model = CoxModel(mock_cox_config)

    # Test Case 1: Base case (male, age 0)
    # LP = (0.05 * 0) + 0.0 = 0
    patient_base = {"age": 0, "sex": "male"}
    lp_base = model.calculate_linear_predictor(patient_base)
    assert lp_base == 0.0

    # Test Case 2: High risk (male, age 10)
    # LP = (0.05 * 10) + 0.0 = 0.5
    patient_high = {"age": 10, "sex": "male"}
    lp_high = model.calculate_linear_predictor(patient_high)
    assert lp_high == 0.5


def test_survival_prediction_logic(mock_cox_config):
    model = CoxModel(mock_cox_config)
    patient = {"age": 10, "sex": "male"}

    # Calculate expected survival
    # LP = 0.5
    # S(t) = S_0(t) ^ exp(LP)
    # S(t) = 0.80 ^ exp(0.5)
    results = model.predict_patient_survival(patient)

    assert 5 in results
    assert 10 in results
    assert 0.0 <= results[5] <= 1.0
    # Higher risk (positive LP) should result in lower survival than baseline
    assert results[5] < 0.80


def test_risk_categorization(mock_cox_config):
    model = CoxModel(mock_cox_config)

    cat_high, desc_high = model.categorize_risk(0.90)
    assert cat_high == "Excellent Prognosis"

    cat_low, desc_low = model.categorize_risk(0.20)
    assert cat_low == "Very Poor Prognosis"
