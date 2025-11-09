"""
Cox Proportional Hazards Model Support
"""

import math
from typing import Any, Dict, Tuple


class CoxModel:
    """
    Cox proportional hazards model for survival prediction.
    Follows same pattern as GastricCancerRiskModel for consistency.
    """

    PROGNOSIS_DESCRIPTIONS = {
        "Excellent Prognosis": "≥85% 5-year survival; resembles KLASS function-preserving cases.",
        "Good Prognosis": "70–85% survival with favorable nodal profile.",
        "Moderate Prognosis": "50–70% survival; careful surveillance recommended.",
        "Poor Prognosis": "30–50% survival; consider intensified adjuvant plans.",
        "Very Poor Prognosis": "<30% survival; aligns with aggressive biology.",
    }

    def __init__(self, config: dict[str, Any]):
        """Initialize from JSON configuration matching your existing pattern."""
        self.config = config
        self.model_id = config.get("id", "cox_model")
        self.name = config.get("name", "Cox Survival Model")
        self.variables = config.get("variables", {})
        self.timepoints = config.get("timepoints", [5, 10])

        baseline = config.get("baseline_survival", {})
        self.baseline_5yr = float(baseline.get("5_year_estimate", 0.65))
        self.baseline_10yr = float(baseline.get("10_year_estimate", 0.55))

    def calculate_linear_predictor(self, patient_data: dict[str, Any]) -> float:
        """
        Calculate Cox linear predictor from patient data.
        Similar to calculate_risk() but returns linear predictor, not probability.
        """
        lp = 0.0

        for var_name, var_config in self.variables.items():
            if var_name not in patient_data:
                continue

            var_type = var_config.get("type")

            if var_type == "categorical":
                patient_category = patient_data[var_name]
                categories = var_config.get("categories", {})
                if patient_category in categories:
                    lp += float(categories[patient_category])

            elif var_type == "continuous":
                patient_value = float(patient_data[var_name])
                coefficient = float(var_config.get("coefficient", 0.0))
                lp += coefficient * patient_value

        return lp

    def calculate_survival(self, patient_data: dict[str, Any]) -> dict[int, float]:
        """
        Calculate survival probabilities at configured timepoints.
        Returns dict like {5: 0.75, 10: 0.62}
        """
        lp = self.calculate_linear_predictor(patient_data)
        survival_5yr = self.baseline_5yr ** math.exp(lp)
        survival_10yr = self.baseline_10yr ** math.exp(lp)

        survival_5yr = max(0.0, min(1.0, survival_5yr))
        survival_10yr = max(0.0, min(1.0, survival_10yr))
        return {5: survival_5yr, 10: survival_10yr}

    def predict_patient_survival(self, patient_data: dict[str, Any]) -> Dict[int, float]:
        """Public API to compute survival probabilities at configured time points."""
        return self.calculate_survival(patient_data)

    def categorize_risk(self, survival_5yr: float) -> Tuple[str, str]:
        """Return prognosis category plus descriptive text."""
        category = self._survival_category_label(survival_5yr)
        description = self.PROGNOSIS_DESCRIPTIONS.get(
            category, "Survival probability outside configured ranges."
        )
        return category, description

    @staticmethod
    def _survival_category_label(survival_5yr: float) -> str:
        if survival_5yr >= 0.85:
            return "Excellent Prognosis"
        if survival_5yr >= 0.70:
            return "Good Prognosis"
        if survival_5yr >= 0.50:
            return "Moderate Prognosis"
        if survival_5yr >= 0.30:
            return "Poor Prognosis"
        return "Very Poor Prognosis"


def map_patient_to_han2012(row: Any, t_stage: str, n_stage: str) -> dict[str, Any]:
    """
    Map TCGA patient data to Han 2012 format.
    
    Args:
        row: Patient row from dataframe
        t_stage: Already normalized T stage (T1-T4)
        n_stage: Already normalized N stage (N0-N3)
    
    Returns:
        Dictionary with Han 2012 variables
    """
    # Age category
    age = getattr(row, "age", 60.0)
    if age < 40:
        age_cat = "< 40"
    elif age < 50:
        age_cat = "40-49"
    elif age < 60:
        age_cat = "50-59"
    elif age < 70:
        age_cat = "60-69"
    else:
        age_cat = ">= 70"
    
    # Sex (direct mapping)
    sex_raw = str(getattr(row, "Sex", "Male")).strip()
    sex = "female" if sex_raw.lower() in ["female", "f"] else "male"
    
    # Tumor location (imputed - TCGA doesn't have this)
    # Use stage-informed heuristic
    if t_stage in ["T1", "T2"]:
        location = "lower"  # Early cancer often distal
    elif t_stage == "T3":
        location = "middle"
    else:
        location = "upper"  # Advanced more likely proximal
    
    # Depth of invasion (map T stage to Han 2012 categories)
    depth_map = {
        "T1": "submucosa",
        "T2": "proper_muscle",
        "T3": "subserosa",
        "T4": "serosa",
    }
    depth = depth_map.get(t_stage, "proper_muscle")
    
    # Metastatic lymph nodes (estimate from N stage)
    met_ln_map = {
        "N0": "0",
        "N1": "1-2",
        "N2": "3-6",
        "N3": "7-15",
    }
    met_ln_cat = met_ln_map.get(n_stage, "1-2")
    
    # Examined lymph nodes (estimate from N stage)
    examined_map = {
        "N0": 20,
        "N1": 25,
        "N2": 30,
        "N3": 35,
    }
    examined_ln = examined_map.get(n_stage, 25)
    
    return {
        "age": age_cat,
        "sex": sex,
        "location": location,
        "depth_of_invasion": depth,
        "metastatic_lymph_nodes": met_ln_cat,
        "examined_lymph_nodes": examined_ln,
    }
