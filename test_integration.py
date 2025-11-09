"""
Quick Test Script for Dual Model Integration
Tests that all components work with your TCGA data
"""

import sys
import json
from pathlib import Path

print("=" * 70)
print("DUAL MODEL INTEGRATION TEST")
print("=" * 70)

# Test 1: Check file structure
print("\n[1/5] Checking file structure...")
required_files = {
    'models/han2012_jco.json': 'Han 2012 model config',
    'models/cox_model.py': 'Cox model implementation',
    'models/variable_mapper_tcga.py': 'TCGA variable mapper',
    'data/tcga_2018_clinical_data.tsv': 'TCGA data file'
}

all_files_present = True
for filepath, description in required_files.items():
    path = Path(filepath)
    if path.exists():
        print(f"  ✓ {description}: {filepath}")
    else:
        print(f"  ✗ MISSING: {description} at {filepath}")
        all_files_present = False

if not all_files_present:
    print("\n✗ Some files are missing. Please download and place them correctly.")
    sys.exit(1)

# Test 2: Import modules
print("\n[2/5] Testing imports...")
try:
    from models.cox_model import CoxModel
    from models.variable_mapper_tcga import Han2012VariableMapper
    print("  ✓ Cox model imported successfully")
    print("  ✓ Variable mapper imported successfully")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 3: Load Han 2012 model
print("\n[3/5] Loading Han 2012 model...")
try:
    with open('models/han2012_jco.json', 'r') as f:
        config = json.load(f)
    model = CoxModel(config)
    print(f"  ✓ Model loaded: {model.name}")
    print(f"  ✓ Variables: {len(model.variables)}")
except Exception as e:
    print(f"  ✗ Failed to load model: {e}")
    sys.exit(1)

# Test 4: Test variable mapping
print("\n[4/5] Testing variable mapping with sample TCGA patient...")
try:
    test_patient = {
        'age': 65,
        'Sex': 'Female',
        'T_stage': 'T3',
        'N_stage': 'N2',
        'positive_LN': None,
        'total_LN': None,
    }
    
    mapper = Han2012VariableMapper()
    han_patient = mapper.map_patient_from_dict(test_patient)
    
    print("  ✓ Mapping successful")
    print(f"    Age: {test_patient['age']} → {han_patient['age']}")
    print(f"    Sex: {test_patient['Sex']} → {han_patient['sex']}")
    print(f"    T stage: {test_patient['T_stage']} → {han_patient['depth_of_invasion']}")
    print(f"    N stage: {test_patient['N_stage']} → {han_patient['metastatic_lymph_nodes']}")
    print(f"    Examined LNs: {han_patient['examined_lymph_nodes']} (estimated)")
    
except Exception as e:
    print(f"  ✗ Mapping failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test survival prediction
print("\n[5/5] Testing survival prediction...")
try:
    survival_probs = model.predict_patient_survival(han_patient)
    
    print("  ✓ Prediction successful")
    print(f"    5-year survival: {survival_probs[5] * 100:.1f}%")
    print(f"    10-year survival: {survival_probs[10] * 100:.1f}%")
    
    category, desc = model.categorize_risk(survival_probs[5])
    print(f"    Prognosis: {category}")
    print(f"    Description: {desc}")
    
except Exception as e:
    print(f"  ✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check TCGA data columns
print("\n[6/6] Checking TCGA data columns...")
try:
    import pandas as pd
    df = pd.read_csv('data/tcga_2018_clinical_data.tsv', sep='\t')
    print(f"  ✓ TCGA data loaded: {len(df)} patients")
    
    # Check for required columns
    required_cols = {
        'Diagnosis Age': 'age',
        'Sex': 'sex',
        'American Joint Committee on Cancer Tumor Stage Code': 'T stage',
        'Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code': 'N stage'
    }
    
    for col, description in required_cols.items():
        if col in df.columns:
            n_available = df[col].notna().sum()
            pct = n_available / len(df) * 100
            print(f"  ✓ {description}: {n_available}/{len(df)} ({pct:.1f}%)")
        else:
            print(f"  ✗ {description}: Column '{col}' not found")
    
except Exception as e:
    print(f"  ✗ TCGA data check failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nYou're ready to run the full pipeline:")
print("  python risk_calculator.py --data data/tcga_2018_clinical_data.tsv")
print("\nOr skip survival model:")
print("  python risk_calculator.py --data data/tcga_2018_clinical_data.tsv --skip-survival")
print("=" * 70)
