from pathlib import Path

import pytest

from risk_calculator import *


def test_stage_normalization():
    assert normalize_t_stage("T4a") == "T4"
    assert normalize_n_stage(" n2 ") == "N2"
    assert normalize_t_stage("Stage X") is None


def test_estimation_monotonicity():
    assert estimate_tumor_size("T3") > estimate_tumor_size("T1")
    assert estimate_ln_ratio("N3") > estimate_ln_ratio("N0")


@pytest.mark.parametrize(
    "low_stage, high_stage",
    [
        ({"T_stage": "T1", "N_stage": "N0", "age": 45, "tumor_size_cm": 2.0, "ln_ratio": 0.01},
         {"T_stage": "T4", "N_stage": "N3", "age": 72, "tumor_size_cm": 6.0, "ln_ratio": 0.65}),
    ],
)
def test_logistic_model_outputs(low_stage, high_stage):
    config = load_model_config(DEFAULT_MODEL_CONFIG)
    model = GastricCancerRiskModel(config)
    low_risk = model.calculate_risk(low_stage)
    high_risk = model.calculate_risk(high_stage)
    assert 0.0 < low_risk < high_risk < 1.0
