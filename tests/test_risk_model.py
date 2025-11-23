"""
Unit tests for risk model functionality.
Tests Brier score calculation, model predictions, and survival estimates.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

# Ensure repository root is on sys.path for direct imports when running via pytest.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_calculator import (  # noqa: E402
    DEFAULT_MODEL_CONFIG,
    GastricCancerRiskModel,
    load_model_config,
)
from models.cox_model import CoxModel  # noqa: E402


class TestBrierScore:
    """Tests for calibration metric calculations."""

    def test_brier_score_perfect_predictions(self):
        """Brier score should be 0 for perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])
        score = brier_score_loss(y_true, y_pred)
        assert score == 0.0

    def test_brier_score_random_predictions(self):
        """Brier score should be ~0.25 for constant 0.5 predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        score = brier_score_loss(y_true, y_pred)
        assert 0.24 <= score <= 0.26

    def test_brier_score_bounds(self):
        """Brier score must be between 0 and 1."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0.3, 0.7, 0.2, 0.8, 0.1])
        score = brier_score_loss(y_true, y_pred)
        assert 0.0 <= score <= 1.0


class TestRiskModelFunctionality:
    """Tests for GastricCancerRiskModel predictions."""

    @pytest.fixture
    def model(self):
        """Load the default heuristic model."""
        config = load_model_config(DEFAULT_MODEL_CONFIG)
        return GastricCancerRiskModel(config)

    def test_model_returns_valid_probability(self, model):
        """Risk predictions must be valid probabilities."""
        patient = {
            "T_stage": "T2",
            "N_stage": "N1",
            "age": 60,
            "tumor_size_cm": 3.5,
            "ln_ratio": 0.15,
        }
        risk = model.calculate_risk(patient)
        assert risk is not None
        assert 0.0 <= risk <= 1.0

    def test_model_respects_risk_floor_ceiling(self, model):
        """Extreme cases should be bounded by floor/ceiling."""
        low_risk_patient = {
            "T_stage": "T1",
            "N_stage": "N0",
            "age": 30,
            "tumor_size_cm": 1.0,
            "ln_ratio": 0.0,
        }
        high_risk_patient = {
            "T_stage": "T4",
            "N_stage": "N3",
            "age": 85,
            "tumor_size_cm": 10.0,
            "ln_ratio": 0.9,
        }
        low_risk = model.calculate_risk(low_risk_patient)
        high_risk = model.calculate_risk(high_risk_patient)

        assert low_risk >= model.risk_floor
        assert high_risk <= model.risk_ceiling


class TestSurvivalEstimates:
    """Tests for Han 2012 Cox survival model."""

    @pytest.fixture
    def survival_model(self):
        """Load the Han 2012 survival model."""
        config_path = PROJECT_ROOT / "models" / "han2012_jco.json"
        with open(config_path, "r", encoding="utf-8") as stream:
            config = json.load(stream)
        return CoxModel(config)

    def test_survival_estimates_in_valid_range(self, survival_model):
        """Survival probabilities must be between 0 and 1."""
        patient = {
            "age": "50-59",
            "sex": "male",
            "location": "lower",
            "depth_of_invasion": "proper_muscle",
            "metastatic_lymph_nodes": "1-2",
            "examined_lymph_nodes": 25,
        }
        estimates = survival_model.predict_patient_survival(patient)

        assert 0.0 <= estimates[5] <= 1.0
        assert 0.0 <= estimates[10] <= 1.0

    def test_five_year_exceeds_ten_year_survival(self, survival_model):
        """5-year survival should be >= 10-year survival."""
        patient = {
            "age": "60-69",
            "sex": "female",
            "location": "middle",
            "depth_of_invasion": "subserosa",
            "metastatic_lymph_nodes": "3-6",
            "examined_lymph_nodes": 30,
        }
        estimates = survival_model.predict_patient_survival(patient)

        assert estimates[5] >= estimates[10]

    def test_mean_survival_matches_published_cohort(self, survival_model):
        """
        Mean 5-year survival for typical cohort should approximate
        Han 2012 published value of ~78-80%.
        """
        test_patients = [
            {"age": "50-59", "sex": "male", "location": "lower",
             "depth_of_invasion": "submucosa", "metastatic_lymph_nodes": "0",
             "examined_lymph_nodes": 25},
            {"age": "60-69", "sex": "female", "location": "middle",
             "depth_of_invasion": "proper_muscle", "metastatic_lymph_nodes": "1-2",
             "examined_lymph_nodes": 28},
            {"age": "50-59", "sex": "male", "location": "lower",
             "depth_of_invasion": "subserosa", "metastatic_lymph_nodes": "3-6",
             "examined_lymph_nodes": 30},
        ]

        survivals = [
            survival_model.predict_patient_survival(patient)[5]
            for patient in test_patients
        ]
        mean_survival = sum(survivals) / len(survivals)

        assert 0.70 <= mean_survival <= 0.90
