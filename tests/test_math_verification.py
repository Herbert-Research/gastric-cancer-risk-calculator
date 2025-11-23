from __future__ import annotations

import copy
import math

import pytest

from risk_calculator import DEFAULT_CONFIG_PAYLOAD, GastricCancerRiskModel


def test_gold_standard_manual_calculation_matches_model():
    config = copy.deepcopy(DEFAULT_CONFIG_PAYLOAD)
    model = GastricCancerRiskModel(config)

    patient = {
        "T_stage": "T2",
        "N_stage": "N1",
        "age": 60,
        "tumor_size_cm": 3.0,
        "ln_ratio": 0.2,
    }

    model_probability = model.calculate_risk(patient)

    intercept = config["intercept"]
    t2_weight = config["t_stage_weights"]["T2"]
    n1_weight = config["n_stage_weights"]["N1"]
    age_effect = (60 - config["age_weight"]["pivot"]) * config["age_weight"]["weight"]
    size_effect = 3.0 * config["tumor_size_weight"]["weight"]
    ln_ratio_effect = 0.2 * config["ln_ratio_weight"]

    manual_logit = intercept + t2_weight + n1_weight + age_effect + size_effect + ln_ratio_effect
    manual_probability = 1.0 / (1.0 + math.exp(-manual_logit))

    assert intercept == pytest.approx(-2.25)
    assert t2_weight == pytest.approx(0.9)
    assert n1_weight == pytest.approx(1.1)
    assert age_effect == pytest.approx(0.18)
    assert size_effect == pytest.approx(0.36)
    assert ln_ratio_effect == pytest.approx(0.48)

    assert manual_logit == pytest.approx(0.77, abs=1e-6)
    assert manual_probability == pytest.approx(0.68352, rel=1e-4, abs=1e-6)
    assert model_probability == pytest.approx(manual_probability, rel=1e-9, abs=1e-9)
