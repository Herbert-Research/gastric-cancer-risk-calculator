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
```
Patient A - Early Stage
  Stage: T1N0
  5-Year Recurrence Risk: 16.2%
  Risk Category: Low Risk
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