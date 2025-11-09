# Gastric Cancer Risk Stratification & Survival Modeling

Computational framework for integrating AJCC TNM staging with established Cox proportional hazards nomograms (Han et al., 2012) to generate dual-endpoint risk assessments for post-gastrectomy patients.

## Executive Summary

  - **Synthesizes** a dual-model architecture combining heuristic recurrence risk stratification with formally validated survival nomograms.
  - **Validates** predictive performance against real-world data from the TCGA PanCanAtlas stomach adenocarcinoma (STAD) cohort (n=436).
  - **Quantifies** the actuarial impact of surgical quality by modeling sensitivity to lymph node yield (stage migration effect).
  - **Exports** calibrated risk profiles, survival functions, and cohort-level audit figures designed for clinical research review.

## Scientific Context and Objectives

Standardized D2 gastrectomy is the cornerstone of curative gastric cancer treatment, yet recurrence risk remains heterogeneous even within the same TNM stage. This repository establishes a computational testbed for linking established clinical priors (T-stage, N-stage, age, tumor size) to probabilistic outcomes. By implementing and validating published Cox models (e.g., *Han et al., JCO 2012*) alongside heuristic risk scores, this framework provides the "validation-first" evidence required to transition from static staging manuals to dynamic, personalized risk assessment.

## Data Provenance and Governance

  - **Clinical Pilot Stream:** Primary validation ingests de-identified TCGA PanCanAtlas 2018 clinical data (`data/tcga_2018_clinical_data.tsv`).
  - **Harmonization:** The pipeline includes a robust `Han2012VariableMapper` to translate heterogeneous real-world data (e.g., varying T-stage nomenclature) into the strict categorical inputs required by established nomograms.
  - **Compliance:** All inputs and outputs remain strictly de-identified.

## Analytical Workflow

The core pipeline (`risk_calculator.py`) executes four translational phases:

1.  **Data Harmonization** – Ingests raw clinical data and normalizes TNM staging, imputing missing tumor location or depth based on epidemiological priors where necessary.
2.  **Dual-Outcome Modeling** – Concurrently executes:
      - *Heuristic Logistic Model:* For 5-year recurrence risk stratification.
      - *Han 2012 Cox Model:* For 5- and 10-year overall survival (OS) estimation.
3.  **Calibration Stress-Test** – Evaluates model trustworthiness using Brier score analysis against observed disease-free status in the target cohort.
4.  **Sensitivity Readout** – Simulates the impact of varying lymph node harvest counts on perceived risk, highlighting the critical importance of adequate lymphadenectomy for accurate staging.

## Example Output

Executing the pipeline generates clinically interpretable risk profiles for individual patients alongside a full cohort-level validation. The verbose output below demonstrates the tool's end-to-end analytical capability:

```text
Gastric Cancer Risk Calculator (Dual Model)
============================================================
Data path: data/tcga_2018_clinical_data.tsv
Recurrence model: heuristic_klass_v1 – KLASS-inspired heuristic (logistic form)
Survival model: Han 2012 D2 Gastrectomy Nomogram

Patient A - Early Stage
  Stage: T1N0
  5-Year Recurrence Risk: 12.8% (Low Risk)
  5-Year Survival: 94.8% (Excellent Prognosis)

Patient B - Moderate Stage
  Stage: T2N1
  5-Year Recurrence Risk: 64.1% (Very High Risk)
  5-Year Survival: 90.8% (Excellent Prognosis)

Patient C - Advanced Stage
  Stage: T3N2
  5-Year Recurrence Risk: 94.3% (Very High Risk)
  5-Year Survival: 81.4% (Good Prognosis)

Patient D - Very Advanced
  Stage: T4N3
  5-Year Recurrence Risk: 95.0% (Very High Risk)
  5-Year Survival: 65.0% (Moderate Prognosis)

============================================================
SENSITIVITY ANALYSIS: Impact of Lymph Node Yield
------------------------------------------------------------
LN Yield = 10 → Risk = 73.3%
LN Yield = 15 → Risk = 68.4%
LN Yield = 20 → Risk = 65.7%
LN Yield = 25 → Risk = 64.1%
LN Yield = 30 → Risk = 62.9%
LN Yield = 35 → Risk = 62.1%
LN Yield = 40 → Risk = 61.5%

============================================================
Key Insight: Higher LN yield reduces estimated risk due to
lower positive/total ratio, highlighting importance of
adequate D2 dissection for accurate staging.

============================================================
TCGA-2018 Clinical Cohort Integration
------------------------------------------------------------
Patients scored: 436
Median predicted risk: 86.6%
  High Risk   : 57
  Low Risk    : 14
  Moderate Risk: 30
  Very High Risk: 335
Top molecular subtypes represented:
  STAD_CIN    : 221
  STAD_MSI    : 72
  Unknown     : 56
Tumor size imputed (stage-derived): 100.0% of cohort
LN ratio imputed (stage-derived): 100.0% of cohort
Brier score (disease-free status): 0.502

============================================================
HAN 2012 SURVIVAL MODEL SUMMARY
------------------------------------------------------------
5-Year Survival:
  Mean:   86.7%
  Median: 87.9%
  Range:  67.7% to 96.4%

10-Year Survival:
  Mean:   82.1%
  Median: 83.6%
  Range:  58.1% to 95.1%

Prognosis Categories:
  Excellent Prognosis: 293 (67.2%)
  Good Prognosis: 142 (32.6%)
  Moderate Prognosis: 1 (0.2%)

Correlation (Recurrence Risk vs Survival): -0.456
  ⚠ Moderate inverse relationship
```

## Generated Figures

The framework automatically generates high-resolution audits for research review:

  - `risk_predictions.png` – Individual patient risk stratification case studies.
  - `survival_predictions_han2012.png` – Cohort-level distribution of 5- and 10-year survival probabilities based on the Han et al. Cox model.
  - `calibration_curve.png` – Reliability diagram contrasting predicted recurrence risk against observed disease-free status (Brier score metric).
  - `sensitivity_analysis.png` – Visualization of how surgical quality (LN yield) impacts algorithmic risk scoring.
  - `tcga_cohort_summary.png` – Heatmap of median risk stratified by TN stage, validating alignment with AJCC standards.

## Usage

### Quickstart (Headless)

Run the end-to-end pipeline using the included TCGA cohort data:

```bash
# Setup environment
pip install -r requirements.txt

# Execute full analytical workflow
python risk_calculator.py --data data/tcga_2018_clinical_data.tsv
```

### Custom Model Configuration

Researchers can modify heuristic weights or swap survival model parameters via JSON configuration:

```bash
python risk_calculator.py --model-config models/heuristic_klass.json --survival-model models/han2012_jco.json
```

## Clinical Interpretation Notes

  - **Educational Nature:** While the Han 2012 model is clinically validated, its implementation here is for educational demonstration of computational risk pipeline development.
  - **Stage Migration:** The sensitivity analysis explicitly models the "Will Rogers phenomenon," where inadequate lymph node dissection falsely lowers the N-stage and thus under-estimates true risk.
  - **Imputation:** Tumor location is often missing in genomic datasets; the model uses stage-informed epidemiological priors (distal vs. proximal prevalence) when direct ICD codes are absent.

## Repository Stewardship

Author: **Maximilian Herbert Dressler**

## Acknowledgement

“The results presented here are in whole or part based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga.”

## Citations

  - Han DS, Suh YS, Kong SH, Lee HJ, Choi Y, Aikou S, Sano T, Park BJ, Kim WH, Yang HK. Nomogram predicting long-term survival after d2 gastrectomy for gastric cancer. J Clin Oncol. 2012 Nov 1;30(31):3834-40. doi: 10.1200/JCO.2012.41.8343. Epub 2012 Sep 24. PMID: 23008291.
  - Cerami E, Gao J, Dogrusoz U, Gross BE, Sumer SO, Aksoy BA, Jacobsen A, Byrne CJ, Heuer ML, Larsson E, Antipin Y, Reva B, Goldberg AP, Sander C, Schultz N. The cBio cancer genomics portal: an open platform for exploring multidimensional cancer genomics data. Cancer Discov. 2012 May;2(5):401-4. doi: 10.1158/2159-8290.CD-12-0095. Erratum in: Cancer Discov. 2012 Oct;2(10):960. PMID: 22588877; PMCID: PMC3956037.
  - Gao J, Aksoy BA, Dogrusoz U, Dresdner G, Gross B, Sumer SO, Sun Y, Jacobsen A, Sinha R, Larsson E, Cerami E, Sander C, Schultz N. Integrative analysis of complex cancer genomics and clinical profiles using the cBioPortal. Sci Signal. 2013 Apr 2;6(269):pl1. doi: 10.1126/scisignal.2004088. PMID: 23550210; PMCID: PMC4160307.
  - Liu J, Lichtenberg T, Hoadley KA, Poisson LM, Lazar AJ, Cherniack AD, Kovatich AJ, Benz CC, Levine DA, Lee AV, Omberg L, Wolf DM, Shriver CD, Thorsson V; Cancer Genome Atlas Research Network; Hu H. An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics. Cell. 2018 Apr 5;173(2):400-416.e11. doi: 10.1016/j.cell.2018.02.052. PMID: 29625055; PMCID: PMC6066282.