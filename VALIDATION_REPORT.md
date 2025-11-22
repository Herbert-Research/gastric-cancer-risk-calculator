# Validation Report: Gastric Cancer Risk Models on TCGA Cohort

**Study Type:** External validation (retrospective cohort)  
**Date:** November 2025  
**Author:** Maximilian Herbert Dressler  
**Status:** Educational demonstration - NOT peer-reviewed

---

## TRIPOD Checklist Compliance

This report follows the Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD) guidelines where applicable for an educational demonstration.

| TRIPOD Item | Description | Status | Notes |
|-------------|-------------|--------|-------|
| Title/Abstract | Identify study as external validation | Partial | Declared as external validation and educational demonstration |
| Background | Clinical context and objectives | Complete | Gastric cancer recurrence and survival modeling described |
| Source of Data | Cohort origin, timeframe | Partial | TCGA PanCanAtlas 2018; retrospective genomic cohort |
| Participants | Inclusion/exclusion, baseline characteristics | Partial | TNM-coded stomach adenocarcinoma cases (n=436); genomic selection bias |
| Outcome Definition | How outcomes were measured | Partial | DFS proxy for recurrence; OS not available; survival predictions simulated |
| Predictors | How predictors measured/handled | Complete | TNM staging, age, sex; 100% imputation for size, LN counts, location |
| Sample Size | Rationale and handling | Partial | Full TCGA stomach cohort; no power calculation |
| Missing Data | Imputation approach | Complete | Deterministic stage-informed priors (documented in README) |
| Statistical Analysis | Modeling methods and performance metrics | Complete | Heuristic logistic model; Cox model with estimated S₀(t); Brier score, distribution summaries |
| Results | Performance with CIs where possible | Partial | Point estimates only; no bootstrapping |
| Limitations | Sources of bias and uncertainty | Complete | Outcome mismatch, imputation, baseline survival estimation |
| Interpretation | Clinical implications and next steps | Partial | Educational only; clinical deployment checklist in README |

---

## Executive Summary

**Objective:** Assess performance of Han 2012 survival nomogram and heuristic recurrence model on TCGA gastric adenocarcinoma cohort.  
**Cohort:** TCGA PanCanAtlas stomach adenocarcinoma (STAD), n=436, enriched for advanced stages.  
**Models:** Han 2012 Cox survival nomogram (baseline survival estimated); heuristic logistic recurrence model (KLASS-inspired).  

**Key Findings (post-baseline recalibration):**
- Mean 5-year survival prediction: ~80% (Han 2012) with prognosis categories balanced (~33% Excellent, ~56% Good/Moderate).
- Recurrence model Brier score vs. disease-free status: 0.502 (poor calibration due to endpoint mismatch).
- High-risk distribution: median recurrence risk 86.6%; 77% classified Very High Risk (reflects advanced cohort).
- Correlation (recurrence vs. survival): r ≈ -0.46 (moderate inverse relationship).

**Limitations:**
- 100% variable imputation (tumor location, size, positive/total LN counts) from stage priors.
- Outcome mismatch for recurrence model (disease-free survival vs. recurrence).
- Baseline survival for Han 2012 model estimated, not published.
- Non-representative cohort (genomic selection bias; incomplete follow-up detail).

**Conclusion:** The framework demonstrates computational integration and transparency but is unsuitable for clinical use without full recalibration, prospective validation, and regulatory review.

---

## Methods Overview

- **Design:** External validation on retrospective genomic cohort (TCGA STAD, 2018 release).
- **Predictors:** TNM stage, age, sex; deterministic imputation for tumor size/location and LN metrics when missing.
- **Outcomes:** Disease-free status (proxy) for recurrence calibration; survival predictions are model-generated using estimated S₀(t) values (S₅=0.52, S₁₀=0.43).
- **Performance Metrics:** Risk distribution summaries, Brier score vs. DFS, prognosis category counts, correlation between recurrence and survival outputs. Time-to-event calibration not assessed (DFS/OS not available with full timelines).
- **Software:** `risk_calculator.py` (dual-model pipeline), Matplotlib visualizations; reproducible via `python3 risk_calculator.py --data data/tcga_2018_clinical_data.tsv`.

---

## Sensitivity Analysis: Lymph Node Yield

### Clinical Motivation

Adequate lymph node harvest is critical for accurate gastric cancer staging (Will Rogers phenomenon). Insufficient lymphadenectomy leads to stage migration, underestimated recurrence risk, and potential undertreatment.

### Methodological Approach

- **Test Case:** T2N1 patient with 3 positive nodes (fixed).  
- **Variable:** Total lymph nodes examined (10–40).  
- **Metric:** Predicted 5-year recurrence risk (heuristic model).

### Results

| LN Yield | LN Ratio | Predicted Risk | Change from Baseline |
|----------|----------|----------------|----------------------|
| 10 | 0.30 | 73.3% | Reference |
| 15 | 0.20 | 68.4% | -4.9% |
| 20 | 0.15 | 65.7% | -7.6% |
| 25 | 0.12 | 64.1% | -9.2% |
| 30 | 0.10 | 62.9% | -10.4% |
| 35 | 0.086 | 62.1% | -11.2% |
| 40 | 0.075 | 61.5% | -11.8% |

### Clinical Implications

1. Inadequate dissection penalty: 10 vs. 30 nodes → ~10% absolute risk difference.  
2. Diminishing returns beyond ~25–30 nodes.  
3. D2 standard: AJCC/NCCN recommend ≥16 nodes examined; >25 improves prognostic confidence.

### Model Behavior

The model penalizes inadequate lymph node harvest via the LN ratio term:

```
logit(p) = ... + 2.4 × (positive_LN / total_LN)
```

Lower denominator (inadequate harvest) → Higher ratio → Higher predicted risk.

### Limitations of This Analysis

- Assumes positive node count fixed (clinical scenarios may vary).  
- Does not model stage migration explicitly.  
- Ignores nodal basin distribution (D1 vs. D2 dissection).

### Recommended Quality Metric

Audit LN yield as a surgical quality indicator:
- Target: ≥80% of cases with ≥16 nodes examined.
- Best practice: ≥25 nodes for accurate prognostication.

---

## Recommended Next Steps

1. Re-estimate S₀(t) using local time-to-event data; report calibration plots with confidence intervals.  
2. Replace deterministic imputation with multiple imputation or exclude cases with missing surgical variables when possible.  
3. Collect prospective cohort with standardized recurrence endpoints to reassess Brier score and discrimination (C-index).  
4. Document full TRIPOD item compliance with uncertainty estimates once time-to-event data are available.  
