from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

FIG_RISK_PREDICTIONS = "risk_predictions.png"
FIG_TCGA_SUMMARY = "tcga_cohort_summary.png"
FIG_CALIBRATION = "calibration_curve.png"
FIG_SURVIVAL_PREDICTIONS = "survival_predictions_han2012.png"
FIG_SURVIVAL_VS_RECURRENCE = "survival_vs_recurrence_comparison.png"

CATEGORY_COLORS = {
    "Low Risk": "#2ca02c",
    "Moderate Risk": "#ff7f0e",
    "High Risk": "#d62728",
    "Very High Risk": "#7f0000",
}

T_STAGE_ORDER = ["T1", "T2", "T3", "T4"]
N_STAGE_ORDER = ["N0", "N1", "N2", "N3"]


def finalize_figure(fig: Figure, output_path: Path, show_plots: bool) -> Path:
    """Persist figure to disk and optionally show the interactive window."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show_plots:
        fig.show()
    else:
        plt.close(fig)
    return output_path


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

