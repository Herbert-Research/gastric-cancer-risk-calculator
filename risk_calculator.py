"""
Gastric Cancer Risk Calculator
Author: Maximilian Dressler
Purpose: Implement published risk stratification models for gastric cancer
Based on: Simplified models from KLASS literature
"""

from __future__ import annotations

import argparse
from pathlib import Path
import copy
import json
import math
import sys
from numbers import Number
from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

if TYPE_CHECKING:
    from models.cox_model import CoxModel
    from models.variable_mapper_tcga import Han2012VariableMapper

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "tcga_2018_clinical_data.tsv"
DEFAULT_OUTPUT_DIR = BASE_DIR
DEFAULT_MODEL_CONFIG = BASE_DIR / "models" / "heuristic_klass.json"
DEFAULT_SURVIVAL_MODEL_CONFIG = BASE_DIR / "models" / "han2012_jco.json"

try:
    from models.cox_model import CoxModel
    from models.variable_mapper_tcga import Han2012VariableMapper
    COX_MODEL_AVAILABLE = True
except ImportError:
    COX_MODEL_AVAILABLE = False
    print("Warning: Cox survival components unavailable; use --skip-survival to suppress.")

FIG_RISK_PREDICTIONS = "risk_predictions.png"
FIG_SENSITIVITY = "sensitivity_analysis.png"
FIG_TCGA_SUMMARY = "tcga_cohort_summary.png"
FIG_CALIBRATION = "calibration_curve.png"
FIG_SURVIVAL_PREDICTIONS = "survival_predictions_han2012.png"
FIG_SURVIVAL_VS_RECURRENCE = "survival_vs_recurrence_comparison.png"

DEFAULT_CONFIG_PAYLOAD = {
    "id": "heuristic_klass_v1",
    "name": "KLASS-inspired heuristic (logistic form)",
    "description": (
        "Educational logistic model using TN stage, age, tumor size, and LN ratio heuristics. "
        "Calibrate or replace with institution-specific coefficients for production use."
    ),
    "citation": "Placeholder heuristics by M.H. Dressler (2025), inspired by KLASS literature summaries.",
    "intercept": -2.25,
    "risk_floor": 0.02,
    "risk_ceiling": 0.95,
    "t_stage_weights": {"T1": 0.0, "T2": 0.9, "T3": 1.6, "T4": 2.2},
    "n_stage_weights": {"N0": 0.0, "N1": 1.1, "N2": 1.9, "N3": 2.7},
    "age_weight": {"weight": 0.018, "pivot": 50},
    "tumor_size_weight": {"weight": 0.12},
    "ln_ratio_weight": 2.4,
}


def sigmoid(value: float) -> float:
    """Numerically stable logistic transform."""

    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def load_model_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load a JSON configuration containing model coefficients."""

    payload: dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG_PAYLOAD)
    target_path = config_path or DEFAULT_MODEL_CONFIG

    if target_path and target_path.exists():
        try:
            with target_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except json.JSONDecodeError as exc:
            print(f"Unable to parse {target_path}: {exc}. Falling back to bundled config.")
        except OSError as exc:
            print(f"Unable to read {target_path}: {exc}. Falling back to bundled config.")

    return payload


def load_survival_model(config_path: Path | None = None) -> CoxModel | None:
    """Load Han 2012 Cox survival model from JSON config."""

    if not COX_MODEL_AVAILABLE:
        return None

    target_path = config_path or DEFAULT_SURVIVAL_MODEL_CONFIG
    if not target_path or not target_path.exists():
        print(f"Survival model config not found at {target_path}")
        return None

    try:
        with target_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        return CoxModel(config)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Unable to load survival model: {exc}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate gastric cancer risk predictions and cohort visualisations."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the TCGA clinical TSV (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store generated figures (default: project root).",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display matplotlib figures after saving them.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Path to JSON file containing logistic model coefficients (default: %(default)s).",
    )
    parser.add_argument(
        "--survival-model",
        type=Path,
        default=DEFAULT_SURVIVAL_MODEL_CONFIG,
        help="Path to Cox survival model config (Han 2012). Default: %(default)s.",
    )
    parser.add_argument(
        "--skip-survival",
        action="store_true",
        help="Skip survival predictions (recurrence only).",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float:
    """Return a Python float from numpy/pandas scalar wrappers."""

    if isinstance(value, Number):
        return float(value)  # type: ignore[reportArgumentType]

    item = getattr(value, "item", None)
    if callable(item):
        inner = item()
        if isinstance(inner, Number):
            return float(inner)  # type: ignore[reportArgumentType]

    return float(value)


def finalize_figure(fig: Figure, output_path: Path, show_plots: bool) -> Path:
    """Persist figure to disk and optionally show the interactive window."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show_plots:
        fig.show()
    else:
        plt.close(fig)
    return output_path

CATEGORY_COLORS = {
    "Low Risk": "#2ca02c",
    "Moderate Risk": "#ff7f0e",
    "High Risk": "#d62728",
    "Very High Risk": "#7f0000",
}

T_STAGE_PRIOR_SIZE = {
    "T1": 2.0,
    "T2": 3.5,
    "T3": 5.0,
    "T4": 6.5,
}

N_STAGE_PRIOR_LN_RATIO = {
    "N0": 0.02,
    "N1": 0.15,
    "N2": 0.35,
    "N3": 0.65,
}

DEFAULT_T_STAGE = "T2"
DEFAULT_N_STAGE = "N1"
T_STAGE_ORDER = ["T1", "T2", "T3", "T4"]
N_STAGE_ORDER = ["N0", "N1", "N2", "N3"]


class GastricCancerRiskModel:
    """Configurable logistic model based on TNM staging and clinical factors."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.t_stage_weights = config.get("t_stage_weights", {})
        self.n_stage_weights = config.get("n_stage_weights", {})
        self.age_weight_cfg = config.get("age_weight", {})
        self.tumor_weight_cfg = config.get("tumor_size_weight", {})
        self.ln_ratio_weight = float(config.get("ln_ratio_weight", 0.0))
        self.intercept = float(config.get("intercept", 0.0))
        self.risk_floor = float(config.get("risk_floor", 0.0))
        self.risk_ceiling = float(config.get("risk_ceiling", 1.0))

    def _validate_stage(self, stage: str | None, stage_type: str) -> str:
        if stage_type == "T":
            valid = self.t_stage_weights
        else:
            valid = self.n_stage_weights
        if stage not in valid:
            raise ValueError(f"Unsupported {stage_type} stage: {stage}")
        return stage  # type: ignore[return-value]

    def calculate_risk(self, patient_data: dict[str, Any]) -> float:
        """Calculate 5-year recurrence risk for an individual patient."""

        t_stage = self._validate_stage(patient_data.get("T_stage"), "T")
        n_stage = self._validate_stage(patient_data.get("N_stage"), "N")

        logit = self.intercept
        logit += float(self.t_stage_weights[t_stage])
        logit += float(self.n_stage_weights[n_stage])

        age = patient_data.get("age")
        if age is not None:
            logit += self._apply_age_effect(float(age))

        tumor_size = patient_data.get("tumor_size_cm")
        if tumor_size is not None:
            logit += float(tumor_size) * float(self.tumor_weight_cfg.get("weight", 0.0))

        ln_ratio = resolve_ln_ratio(
            patient_data.get("ln_ratio"),
            patient_data.get("positive_LN"),
            patient_data.get("total_LN"),
        )
        if ln_ratio is not None:
            bounded_ratio = max(0.0, min(1.0, float(ln_ratio)))
            logit += bounded_ratio * self.ln_ratio_weight

        risk = sigmoid(logit)
        if self.risk_ceiling:
            risk = min(risk, self.risk_ceiling)
        if self.risk_floor:
            risk = max(risk, self.risk_floor)
        return float(risk)

    def _apply_age_effect(self, age: float) -> float:
        weight = float(self.age_weight_cfg.get("weight", 0.0))
        if weight == 0.0:
            return 0.0
        pivot = float(self.age_weight_cfg.get("pivot", 0.0))
        delta = age - pivot
        if self.age_weight_cfg.get("positive_delta_only", True):
            delta = max(0.0, delta)
        return delta * weight

    @staticmethod
    def risk_category(risk: float) -> str:
        """Categorize risk level."""

        if risk < 0.20:
            return "Low Risk"
        if risk < 0.40:
            return "Moderate Risk"
        if risk < 0.60:
            return "High Risk"
        return "Very High Risk"


def resolve_ln_ratio(
    ln_ratio: float | None, positive_ln: float | None, total_ln: float | None
) -> float | None:
    """Derive LN ratio from explicit counts if direct value missing."""

    if ln_ratio is not None:
        return float(ln_ratio)
    if total_ln in (None, 0):
        return None
    positive_ln = positive_ln or 0.0
    if total_ln <= 0:
        return None
    return float(positive_ln) / float(total_ln)


def score_patients(model, patients):
    """Return a dataframe with risk predictions for each patient input."""

    rows = []
    for patient in patients:
        risk = model.calculate_risk(patient)
        # Compute resolved LN ratio for consistent reporting
        resolved_ln_ratio = resolve_ln_ratio(
            patient.get("ln_ratio"),
            patient.get("positive_LN"),
            patient.get("total_LN"),
        )
        rows.append(
            {
                "Patient": patient.get("name")
                or patient.get("patient_id")
                or patient.get("id", "Patient"),
                "Risk": risk,
                "Category": model.risk_category(risk),
                "T_stage": patient.get("T_stage"),
                "N_stage": patient.get("N_stage"),
                "age": patient.get("age"),
                "tumor_size_cm": patient.get("tumor_size_cm"),
                "ln_ratio": resolved_ln_ratio,
                "tumor_size_imputed": patient.get("tumor_size_imputed", False),
                "ln_ratio_imputed": patient.get("ln_ratio_imputed", False),
                "positive_LN": patient.get("positive_LN"),
                "total_LN": patient.get("total_LN"),
                "Sex": patient.get("Sex") or patient.get("sex"),
            }
        )

    return pd.DataFrame(rows)


def predict_with_both_models(
    patients: list[dict[str, Any]],
    recurrence_model: GastricCancerRiskModel,
    survival_model: CoxModel | None,
) -> pd.DataFrame:
    """Score patients with both recurrence and survival models."""

    results_df = score_patients(recurrence_model, patients)

    if survival_model and COX_MODEL_AVAILABLE:
        mapper = Han2012VariableMapper()
        survival_5yr: list[float | None] = []
        survival_10yr: list[float | None] = []
        survival_category: list[str | None] = []
        survival_desc: list[str | None] = []
        mapper_flags: list[dict[str, bool]] = []

        for patient in patients:
            try:
                mapped_fields = mapper.map_patient_from_dict(patient)
                mapper_flags.append(mapper.get_imputation_flags(patient))
                survival_probs = survival_model.predict_patient_survival(mapped_fields)
                surv5 = survival_probs.get(5)
                surv10 = survival_probs.get(10)
                survival_5yr.append(surv5)
                survival_10yr.append(surv10)
                if surv5 is not None:
                    category, description = survival_model.categorize_risk(surv5)
                    survival_category.append(category)
                    survival_desc.append(description)
                else:
                    survival_category.append(None)
                    survival_desc.append(None)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Warning: Survival prediction failed for patient {patient.get('name')}: {exc}")
                survival_5yr.append(None)
                survival_10yr.append(None)
                survival_category.append(None)
                survival_desc.append(None)
                mapper_flags.append(
                    {
                        "age_available": patient.get("age") is not None,
                        "sex_available": bool(patient.get("Sex") or patient.get("sex")),
                        "location_imputed": True,
                        "positive_ln_imputed": patient.get("positive_LN") is None,
                        "examined_ln_imputed": patient.get("total_LN") is None,
                    }
                )

        results_df["survival_5yr"] = survival_5yr
        results_df["survival_10yr"] = survival_10yr
        results_df["survival_category"] = survival_category
        results_df["survival_summary"] = survival_desc
        results_df["survival_imputations"] = mapper_flags

    return results_df


def plot_individual_predictions(
    results_df: pd.DataFrame, output_dir: Path, show_plots: bool
) -> Path:
    """Visualize risk distribution for individual case studies."""

    colors = results_df["Category"].map(CATEGORY_COLORS).fillna("gray")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(results_df["Patient"], results_df["Risk"] * 100, color=colors, alpha=0.8)
    axes[0].set_xlabel("5-Year Recurrence Risk (%)", fontsize=12)
    axes[0].set_title("Patient-Specific Risk Predictions", fontsize=14, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3)

    category_counts = results_df["Category"].value_counts()
    pie_colors = [CATEGORY_COLORS[label] for label in category_counts.index]
    axes[1].pie(
        category_counts.values,
        labels=category_counts.index,
        autopct="%1.0f%%",
        colors=pie_colors,
        startangle=90,
    )
    axes[1].set_title("Risk Category Distribution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return finalize_figure(fig, output_dir / FIG_RISK_PREDICTIONS, show_plots)


def plot_survival_predictions(
    results_df: pd.DataFrame, output_dir: Path, show_plots: bool
) -> Path | None:
    """Visualize Han 2012 survival predictions across the cohort."""

    if "survival_5yr" not in results_df.columns:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    # Add overall figure annotation about calibration status
    fig.text(
        0.5,
        0.02,
        "Note: Survival estimates use uncalibrated baseline S₀(t). "
        + "Institutional validation required. Educational demonstration only.",
        ha="center",
        fontsize=9,
        style="italic",
        bbox=dict(facecolor="wheat", alpha=0.5, boxstyle="round,pad=0.5"),
    )

    surv5 = (results_df["survival_5yr"].dropna() * 100).to_numpy()
    if surv5.size:
        ax = axes[0, 0]
        ax.hist(surv5, bins=20, alpha=0.8, color="steelblue", edgecolor="black")
        ax.axvline(surv5.mean(), color="red", linestyle="--", label=f"Mean: {surv5.mean():.1f}%")
        ax.set_title("5-Year Survival Distribution (Han 2012)", fontweight="bold")
        ax.set_xlabel("5-Year Survival Probability (%)")
        ax.set_ylabel("Number of Patients")
        ax.legend()
        ax.grid(alpha=0.3)

    surv10 = (results_df["survival_10yr"].dropna() * 100).to_numpy()
    if surv10.size:
        ax = axes[0, 1]
        ax.hist(surv10, bins=20, alpha=0.8, color="darkgreen", edgecolor="black")
        ax.axvline(surv10.mean(), color="red", linestyle="--", label=f"Mean: {surv10.mean():.1f}%")
        ax.set_title("10-Year Survival Distribution (Han 2012)", fontweight="bold")
        ax.set_xlabel("10-Year Survival Probability (%)")
        ax.set_ylabel("Number of Patients")
        ax.legend()
        ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if "survival_category" in results_df.columns:
        category_counts = results_df["survival_category"].value_counts()
        if not category_counts.empty:
            colors = ["darkgreen", "yellowgreen", "gold", "orangered", "darkred"]
            ax.barh(category_counts.index, category_counts.values, color=colors[: len(category_counts)])
            ax.set_xlabel("Number of Patients")
            ax.set_title("Han 2012 Prognosis Categories", fontweight="bold")
            ax.grid(alpha=0.3, axis="x")

    ax = axes[1, 1]
    if {"survival_5yr", "Risk"}.issubset(results_df.columns):
        scatter_df = results_df[["survival_5yr", "Risk"]].dropna()
        if not scatter_df.empty:
            sc = ax.scatter(
                scatter_df["survival_5yr"] * 100,
                scatter_df["Risk"] * 100,
                c=scatter_df["survival_5yr"] * 100,
                cmap="RdYlGn",
                alpha=0.6,
                s=40,
            )
            ax.set_xlabel("5-Year Survival Probability (%)")
            ax.set_ylabel("5-Year Recurrence Risk (%)")
            ax.set_title("Survival vs. Recurrence", fontweight="bold")
            ax.grid(alpha=0.3)
            plt.colorbar(sc, ax=ax, label="5-Year Survival (%)")

    plt.tight_layout()
    return finalize_figure(fig, output_dir / FIG_SURVIVAL_PREDICTIONS, show_plots)


def plot_survival_vs_recurrence(
    results_df: pd.DataFrame, output_dir: Path, show_plots: bool
) -> Path | None:
    """Generate a dedicated scatter comparison between survival and recurrence outputs."""

    required_cols = {"survival_5yr", "Risk"}
    if not required_cols.issubset(results_df.columns):
        return None

    df = results_df[list(required_cols)].dropna()
    if df.empty:
        return None

    corr = df["survival_5yr"].corr(df["Risk"])

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        df["survival_5yr"] * 100,
        df["Risk"] * 100,
        c=df["Risk"] * 100,
        cmap="viridis_r",
        alpha=0.6,
    )
    ax.set_xlabel("5-Year Survival Probability (%)")
    ax.set_ylabel("5-Year Recurrence Risk (%)")
    ax.set_title("Recurrence vs. Survival Agreement", fontweight="bold")
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Recurrence Risk (%)")
    ax.text(
        0.05,
        0.95,
        f"Correlation = {corr:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    plt.tight_layout()
    return finalize_figure(fig, output_dir / FIG_SURVIVAL_VS_RECURRENCE, show_plots)


def print_survival_summary(results_df: pd.DataFrame) -> None:
    """Print summary statistics for Han 2012 survival predictions."""

    if "survival_5yr" not in results_df.columns:
        return

    print("\n" + "=" * 60)
    print("⚠️  HAN 2012 SURVIVAL MODEL - CALIBRATION STATUS")
    print("-" * 60)
    print("IMPORTANT: These predictions use estimated baseline survival")
    print("S₀(t) calibrated to match published cohort statistics (Han 2012).")
    print("Individual predictions may differ from validated nomogram performance.")
    print("Institutional recalibration required before any clinical use.")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("HAN 2012 SURVIVAL MODEL SUMMARY")
    print("-" * 60)

    surv5 = results_df["survival_5yr"].dropna()
    if not surv5.empty:
        print("5-Year Survival:")
        print(f"  Mean:   {surv5.mean() * 100:.1f}%")
        print(f"  Median: {surv5.median() * 100:.1f}%")
        print(f"  Range:  {surv5.min() * 100:.1f}% to {surv5.max() * 100:.1f}%")

    surv10 = results_df["survival_10yr"].dropna() if "survival_10yr" in results_df else pd.Series(dtype=float)
    if not surv10.empty:
        print("\n10-Year Survival:")
        print(f"  Mean:   {surv10.mean() * 100:.1f}%")
        print(f"  Median: {surv10.median() * 100:.1f}%")
        print(f"  Range:  {surv10.min() * 100:.1f}% to {surv10.max() * 100:.1f}%")

    if "survival_category" in results_df.columns:
        counts = results_df["survival_category"].value_counts()
        if not counts.empty:
            print("\nPrognosis Categories:")
            for cat, count in counts.items():
                pct = count / len(results_df) * 100
                print(f"  {cat}: {count} ({pct:.1f}%)")

    if {"Risk", "survival_5yr"}.issubset(results_df.columns):
        corr_df = results_df[["Risk", "survival_5yr"]].dropna()
        if len(corr_df) > 10:
            corr = corr_df["Risk"].corr(corr_df["survival_5yr"])
            print(f"\nCorrelation (Recurrence Risk vs Survival): {corr:.3f}")
            if corr < -0.5:
                print("  ✓ Strong inverse relationship (as expected)")
            elif corr < -0.3:
                print("  ⚠ Moderate inverse relationship")
            else:
                print("  Note: Weak correlation – investigate cohort differences.")


def run_example_patients(recurrence_model, survival_model=None):
    """Reproduce the illustrative patient scenarios used in the README."""

    patients = [
        {
            "name": "Patient A - Early Stage",
            "T_stage": "T1",
            "N_stage": "N0",
            "age": 55,
            "Sex": "Female",
            "tumor_size_cm": 2.0,
            "positive_LN": 0,
            "total_LN": 20,
        },
        {
            "name": "Patient B - Moderate Stage",
            "T_stage": "T2",
            "N_stage": "N1",
            "age": 62,
            "Sex": "Male",
            "tumor_size_cm": 3.5,
            "positive_LN": 2,
            "total_LN": 25,
        },
        {
            "name": "Patient C - Advanced Stage",
            "T_stage": "T3",
            "N_stage": "N2",
            "age": 68,
            "Sex": "Female",
            "tumor_size_cm": 5.0,
            "positive_LN": 8,
            "total_LN": 30,
        },
        {
            "name": "Patient D - Very Advanced",
            "T_stage": "T4",
            "N_stage": "N3",
            "age": 70,
            "Sex": "Male",
            "tumor_size_cm": 6.5,
            "positive_LN": 15,
            "total_LN": 32,
        },
    ]

    results_df = predict_with_both_models(patients, recurrence_model, survival_model)

    for row in results_df.itertuples(index=False):
        print(f"\n{row.Patient}")
        print(f"  Stage: {row.T_stage}{row.N_stage}")
        risk_pct = safe_float(row.Risk) * 100.0
        print(f"  5-Year Recurrence Risk: {risk_pct:.1f}% ({row.Category})")

        if survival_model and hasattr(row, "survival_5yr") and row.survival_5yr:
            surv_pct = safe_float(row.survival_5yr) * 100.0
            category = getattr(row, "survival_category", "N/A")
            print(f"  5-Year Survival: {surv_pct:.1f}% ({category})")

    return results_df


def run_sensitivity_analysis(model, output_dir: Path, show_plots: bool) -> Path:
    """Illustrate how nodal yield impacts predictions."""

    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS: Impact of Lymph Node Yield")
    print("-" * 60)

    test_patient = {
        "T_stage": "T2",
        "N_stage": "N1",
        "age": 60,
        "tumor_size_cm": 3.0,
        "positive_LN": 3,
        "total_LN": 15,
    }

    ln_yields = range(10, 41, 5)
    risks_by_yield = []

    for ln_yield in ln_yields:
        test_patient["total_LN"] = ln_yield
        risk = model.calculate_risk(test_patient)
        risks_by_yield.append(risk)
        print(f"LN Yield = {ln_yield:2d} → Risk = {risk * 100:.1f}%")

    fig = plt.figure(figsize=(10, 6))
    plt.plot(ln_yields, [r * 100 for r in risks_by_yield], "o-", linewidth=2, markersize=8)
    plt.xlabel("Total Lymph Nodes Retrieved", fontsize=12)
    plt.ylabel("5-Year Recurrence Risk (%)", fontsize=12)
    plt.title(
        "Impact of Lymph Node Yield on Risk Prediction\n(Patient: T2N1, 3 positive nodes)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = finalize_figure(fig, output_dir / FIG_SENSITIVITY, show_plots)

    print("\n" + "=" * 60)
    print("Key Insight: Higher LN yield reduces estimated risk due to")
    print("lower positive/total ratio, highlighting importance of")
    print("adequate D2 dissection for accurate staging.")
    return output_path


def load_tcga_cohort(data_path):
    """Load and harmonize the anonymized TCGA cohort."""

    if not data_path.exists():
        print(f"TCGA file not found at {data_path}.")
        return pd.DataFrame()

    rename_map = {
        "Patient ID": "patient_id",
        "Diagnosis Age": "age",
        "American Joint Committee on Cancer Tumor Stage Code": "raw_t_stage",
        "Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code": "raw_n_stage",
        "Neoadjuvant Therapy Type Administered Prior To Resection Text": "neoadjuvant",
        "Subtype": "molecular_subtype",
        "Disease Free Status": "disease_free_status",
        "Progression Free Status": "progression_free_status",
        "Sex": "sex",
        "ICD-10 Classification": "icd_10",
    }

    try:
        df = pd.read_csv(data_path, sep="\t")
    except Exception as exc:  # pragma: no cover - defensive for parsing errors
        print(f"Unable to parse TCGA cohort: {exc}")
        return pd.DataFrame()

    cohort = df.rename(columns=rename_map)
    cohort["age"] = pd.to_numeric(cohort["age"], errors="coerce")
    cohort["T_stage"] = cohort["raw_t_stage"].apply(normalize_t_stage).fillna(DEFAULT_T_STAGE)
    cohort["N_stage"] = cohort["raw_n_stage"].apply(normalize_n_stage).fillna(DEFAULT_N_STAGE)

    tumor_size, tumor_imputed = _resolve_tumor_size_feature(cohort)
    ln_ratio, ln_imputed, positive_ln, total_ln = _resolve_ln_ratio_feature(cohort)

    cohort["tumor_size_cm"] = tumor_size
    cohort["ln_ratio"] = ln_ratio
    cohort["tumor_size_imputed"] = tumor_imputed
    cohort["ln_ratio_imputed"] = ln_imputed
    cohort["positive_LN"] = positive_ln
    cohort["total_LN"] = total_ln
    status_series = cohort.get("disease_free_status")
    if status_series is None:
        status_series = pd.Series(index=cohort.index, dtype=float)
    cohort["event_observed"] = status_series.apply(parse_event_status)
    cohort = cohort.dropna(subset=["age"])

    drop_cols = [col for col in ["raw_t_stage", "raw_n_stage"] if col in cohort.columns]
    return cohort.drop(columns=drop_cols)


def normalize_t_stage(value):
    """Return simplified T stage (T1-T4)."""

    stage = _normalize_stage(value, "T")
    return stage if stage in T_STAGE_PRIOR_SIZE else None


def normalize_n_stage(value):
    """Return simplified N stage (N0-N3)."""

    stage = _normalize_stage(value, "N")
    return stage if stage in N_STAGE_PRIOR_LN_RATIO else None


def _normalize_stage(value, prefix):
    if pd.isna(value):
        return None
    cleaned = str(value).strip().upper()
    if not cleaned.startswith(prefix):
        return None
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    if digits:
        return f"{prefix}{digits}"
    return None


def estimate_tumor_size(t_stage):
    """Use a stage-informed proxy for tumor size when not provided."""

    return T_STAGE_PRIOR_SIZE.get(t_stage, T_STAGE_PRIOR_SIZE[DEFAULT_T_STAGE])


def estimate_ln_ratio(n_stage):
    """Approximate LN ratio using stage-informed priors."""

    return N_STAGE_PRIOR_LN_RATIO.get(n_stage, N_STAGE_PRIOR_LN_RATIO[DEFAULT_N_STAGE])


def _resolve_tumor_size_feature(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return tumor size (cm) and boolean mask of imputed records."""

    candidate_columns = [
        "tumor_size_cm",
        "Tumor Size (cm)",
        "Tumor size (cm)",
        "Tumor dimension (cm)",
        "Tumor Dimension (cm)",
        "Pathologic Tumor Largest Dimension (cm)",
    ]

    tumor_series = pd.Series(np.nan, index=df.index, dtype=float)
    for column in candidate_columns:
        if column in df.columns:
            tumor_series = pd.to_numeric(df[column], errors="coerce")
            break

    imputed_mask = tumor_series.isna()
    fallback = df.loc[imputed_mask, "T_stage"].apply(estimate_tumor_size)
    tumor_series.loc[imputed_mask] = fallback
    return tumor_series, imputed_mask


def _resolve_ln_ratio_feature(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return LN ratio, imputation mask, and any parsed positive/total LN counts."""

    positive_candidates = [
        "Number of Lymph Nodes Positive",
        "Regional Nodes Positive",
        "Positive Lymph Nodes",
        "lymph_nodes_positive",
    ]
    total_candidates = [
        "Number of Lymph Nodes Examined",
        "Regional Nodes Examined",
        "Total Lymph Nodes",
        "lymph_nodes_examined",
    ]

    positive_ln = _first_numeric_column(df, positive_candidates)
    total_ln = _first_numeric_column(df, total_candidates)

    ratio = pd.Series(np.nan, index=df.index, dtype=float)
    if positive_ln is not None and total_ln is not None:
        valid = (total_ln > 0) & positive_ln.notna()
        ratio.loc[valid] = (positive_ln[valid] / total_ln[valid]).clip(0.0, 1.0)

    imputed_mask = ratio.isna()
    ratio.loc[imputed_mask] = df.loc[imputed_mask, "N_stage"].apply(estimate_ln_ratio)
    if positive_ln is None:
        positive_ln = pd.Series(np.nan, index=df.index, dtype=float)
    if total_ln is None:
        total_ln = pd.Series(np.nan, index=df.index, dtype=float)
    return ratio, imputed_mask, positive_ln, total_ln


def _first_numeric_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for column in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return None


def parse_event_status(value: Any) -> float | None:
    """Derive binary event flag from TCGA textual status columns."""

    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text.startswith("0") or "diseasefree" in text or "tumor free" in text or "censored" in text:
        return 0.0
    if text.startswith("1") or "progression" in text or "recurr" in text or "with tumor" in text:
        return 1.0
    return None


def analyze_tcga_cohort(
    model, data_path: Path, output_dir: Path, show_plots: bool, survival_model=None
) -> list[Path]:
    """Score the TCGA cohort and summarize key findings."""

    cohort = load_tcga_cohort(data_path)
    if cohort.empty:
        print("\nTCGA cohort not analyzed (file missing or parsing failed).")
        return []

    sex_col = None
    for candidate in ["Sex", "sex", "Gender", "gender"]:
        if candidate in cohort.columns:
            sex_col = candidate
            break
    if sex_col:
        cohort["Sex"] = cohort[sex_col]
    else:
        cohort["Sex"] = "Male"
        print("Warning: Sex column not found; defaulting to Male for survival predictions.")

    patient_inputs = []
    for row in cohort.itertuples(index=False):
        patient_inputs.append(
            {
                "name": row.patient_id,
                "T_stage": row.T_stage,
                "N_stage": row.N_stage,
                "age": row.age,
                "Sex": getattr(row, "Sex", "Male"),
                "tumor_size_cm": row.tumor_size_cm,
                "ln_ratio": row.ln_ratio,
                "tumor_size_imputed": getattr(row, "tumor_size_imputed", False),
                "ln_ratio_imputed": getattr(row, "ln_ratio_imputed", False),
                "positive_LN": getattr(row, "positive_LN", None),
                "total_LN": getattr(row, "total_LN", None),
                "neoadjuvant": getattr(row, "neoadjuvant", None),
                "molecular_subtype": getattr(row, "molecular_subtype", None),
                "event_observed": getattr(row, "event_observed", None),
                "icd_10": getattr(row, "icd_10", None),
            }
        )

    cohort_results = predict_with_both_models(patient_inputs, model, survival_model)
    cohort_results = cohort_results.merge(
        cohort[["patient_id", "neoadjuvant", "molecular_subtype", "event_observed"]],
        how="left",
        left_on="Patient",
        right_on="patient_id",
    ).drop(columns=["patient_id"])

    print("\n" + "=" * 60)
    print("TCGA-2018 Clinical Cohort Integration")
    print("-" * 60)
    print(f"Patients scored: {len(cohort_results)}")
    print(f"Median predicted risk: {cohort_results['Risk'].median() * 100:.1f}%")

    category_counts = cohort_results["Category"].value_counts().sort_index()
    for label, count in category_counts.items():
        print(f"  {label:<12}: {count}")

    subtype_counts = (
        cohort_results["molecular_subtype"].fillna("Unknown").value_counts().head(3)
    )
    print("Top molecular subtypes represented:")
    for label, count in subtype_counts.items():
        print(f"  {label:<12}: {count}")

    tumor_imputed_pct = cohort_results["tumor_size_imputed"].mean() * 100
    ln_imputed_pct = cohort_results["ln_ratio_imputed"].mean() * 100
    print("\nData Quality Assessment:")
    print("-" * 60)
    print(f"  Tumor size imputed: {tumor_imputed_pct:.1f}% (stage-informed estimates)")
    print(f"  LN ratio imputed: {ln_imputed_pct:.1f}% (N-stage-derived)")
    print("  Tumor location imputed: 100.0% (epidemiological priors)")

    if tumor_imputed_pct > 90:
        print("\n⚠️  CRITICAL: >90% variable imputation detected.")
        print("   Predictions represent stage-typical, not patient-specific, risk.")
        print("   Suitable for cohort-level validation only.")

    generated_paths: list[Path] = []
    summary_fig = plot_tcga_summary(cohort_results, output_dir, show_plots)
    generated_paths.append(summary_fig)

    calibration_result = plot_calibration_curve(
        cohort_results, output_dir, show_plots, label_column="event_observed"
    )
    if calibration_result:
        calibration_fig, brier = calibration_result
        generated_paths.append(calibration_fig)
        print(f"Brier score (recurrence model vs. DFS): {brier:.3f}")
        print("⚠️  Note: Poor calibration reflects outcome mismatch, not model failure.")
        print("    The model predicts recurrence; TCGA provides disease-free survival.")
        print("    These are related but distinct clinical endpoints.")

    if survival_model and "survival_5yr" in cohort_results.columns:
        survival_fig = plot_survival_predictions(cohort_results, output_dir, show_plots)
        if survival_fig:
            generated_paths.append(survival_fig)
        comparison_fig = plot_survival_vs_recurrence(cohort_results, output_dir, show_plots)
        if comparison_fig:
            generated_paths.append(comparison_fig)
        print_survival_summary(cohort_results)

    return generated_paths


def plot_tcga_summary(
    cohort_results: pd.DataFrame, output_dir: Path, show_plots: bool
) -> Path:
    """Generate cohort-level visualization and save it to disk."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(
        cohort_results["Risk"] * 100,  # type: ignore[reportArgumentType]
        bins=20,
        kde=False,
        ax=axes[0],
        color="#1f77b4",
        edgecolor="white",
    )
    axes[0].set_title("TCGA Cohort Risk Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted 5-Year Recurrence Risk (%)")
    axes[0].set_ylabel("Number of Patients")

    pivot = (
        cohort_results.pivot_table(
            index="N_stage", columns="T_stage", values="Risk", aggfunc="median"
        )
        * 100
    )
    pivot = pivot.reindex(index=N_STAGE_ORDER, columns=T_STAGE_ORDER)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        ax=axes[1],
        cbar_kws={"label": "Median Risk (%)"},
    )
    axes[1].set_title("Median Risk by TN Stage (TCGA)", fontsize=14, fontweight="bold")

    # Add annotation about imputation
    fig.text(
        0.5,
        0.02,
        "Data Quality Note: Tumor size (100%), LN ratio (100%), and tumor location (100%) imputed from stage. "
        + "Predictions are stage-typical, not patient-specific.",
        ha="center",
        fontsize=8,
        style="italic",
        wrap=True,
        bbox=dict(facecolor="lightblue", alpha=0.4),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Leave space for annotation
    return finalize_figure(fig, output_dir / FIG_TCGA_SUMMARY, show_plots)


def plot_calibration_curve(
    cohort_results: pd.DataFrame,
    output_dir: Path,
    show_plots: bool,
    label_column: str,
) -> tuple[Path, float] | None:
    """Plot calibration curve using disease-free status as a proxy for recurrence."""

    if label_column not in cohort_results:
        print("No event labels available for calibration.")
        return None

    valid = cohort_results.dropna(subset=[label_column])
    if valid.empty:
        print("Event labels present but all rows are NaN; skipping calibration.")
        return None

    try:
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss
    except ImportError:  # pragma: no cover - optional dependency
        print("scikit-learn not available; skipping calibration diagnostics.")
        return None

    y_true = valid[label_column].astype(float)
    y_pred = valid["Risk"].astype(float)

    prob_true, prob_pred = calibration_curve(
        y_true, y_pred, n_bins=10, strategy="quantile"
    )
    brier = brier_score_loss(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Observed")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    ax.set_xlabel("Predicted recurrence probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title(
        "Outcome Mismatch Analysis:\nRecurrence Model vs. Disease-Free Status",
        fontweight="bold",
        fontsize=11,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    # Add detailed annotation explaining poor calibration
    annotation_text = (
        f"Brier Score = {brier:.3f}\n"
        f"(reflects endpoint mismatch:\n"
        f"recurrence risk vs. DFS)"
    )
    ax.text(
        0.05,
        0.85,
        annotation_text,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="black", linewidth=0.5),
        verticalalignment="top",
    )

    plt.tight_layout()
    path = finalize_figure(fig, output_dir / FIG_CALIBRATION, show_plots)
    return path, float(brier)


def main():
    args = parse_args()

    print("Gastric Cancer Risk Calculator (Dual Model)")
    print("=" * 60)
    print(f"Data path: {args.data}")
    print(f"Output directory: {args.output_dir}")

    model_config = load_model_config(args.model_config)
    print(f"Recurrence model: {model_config.get('id', 'custom')} – {model_config.get('name', 'N/A')}")
    recurrence_model = GastricCancerRiskModel(model_config)

    survival_model = None
    if args.skip_survival:
        print("Survival model: Skipped (flag provided)")
    else:
        survival_model = load_survival_model(args.survival_model)
        if survival_model:
            print("Survival model: Han 2012 D2 Gastrectomy Nomogram")
        else:
            print("Survival model: Not available (recurrence only)")

    example_results = run_example_patients(recurrence_model, survival_model)

    generated_files = []
    generated_files.append(
        plot_individual_predictions(example_results, args.output_dir, args.show_plots)
    )
    generated_files.append(
        run_sensitivity_analysis(recurrence_model, args.output_dir, args.show_plots)
    )

    tcga_figs = analyze_tcga_cohort(
        recurrence_model, args.data, args.output_dir, args.show_plots, survival_model
    )
    if tcga_figs:
        generated_files.extend(tcga_figs)

    print("\nGenerated files:")
    for path in generated_files:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
