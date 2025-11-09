# Gastric Cancer Risk Calculator

Implementation of clinical risk models for predicting gastric cancer recurrence after surgery.

## Overview

This calculator implements a simplified risk stratification model incorporating:
- TNM staging (T and N stage)
- Patient age
- Tumor size
- Lymph node ratio (positive/total)

## Educational Purpose

This is a **simplified educational implementation** for understanding risk modeling principles. Real clinical applications require:
- Validation on institutional data
- External validation cohorts
- Prospective evaluation
- Regulatory approval

## Features

- **Individual risk calculation** for specific patients
- **Risk stratification** (Low/Moderate/High/Very High)
- **Sensitivity analysis** showing impact of lymph node yield
- **Visualization** of risk distributions

## Clinical Relevance

Understanding which factors drive recurrence risk is essential for:
- Personalized surveillance protocols
- Adjuvant therapy decisions
- Station-specific dissection extent decisions (my PhD focus)

## Usage
```bash
python risk_calculator.py
```

## Sample Output

### Complete Program Output

```
Gastric Cancer Risk Calculator (Dual Model)
============================================================
Data path: C:\Users\m4rti\Documents\GitHub\gastric-cancer-risk-calculator\data\tcga_2018_clinical_data.tsv
Output directory: C:\Users\m4rti\Documents\GitHub\gastric-cancer-risk-calculator
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

Generated files:
  - risk_predictions.png
  - sensitivity_analysis.png
  - tcga_cohort_summary.png
  - calibration_curve.png
  - survival_predictions_han2012.png
  - survival_vs_recurrence_comparison.png
```

## Author

Maximilian Dressler  
PhD Research Focus: Station-specific risk prediction for KLASS gastrectomy  
Clinical Training: Seoul National University Hospital (2025)

## Disclaimer

Not for clinical use. For research and educational purposes only.
```

---

## **FINAL SETUP:**

### **Update Your CV (5 minutes)**

**Change this:**
```
Version Control: Git, GitHub (github.com/[username]/medical-ai-validation)
```

**To this:**
```
Version Control: Git, GitHub
  • medical-ai-validation: Survival analysis and calibration tools
  • gastric-cancer-staging-visualization: KLASS anatomy and staging
  • gastric-cancer-risk-calculator: Risk stratification models
```

Or simpler:
```
Version Control: Git, GitHub (github.com/[yourusername] - 3 repositories 
demonstrating survival analysis, clinical visualization, and risk modeling)