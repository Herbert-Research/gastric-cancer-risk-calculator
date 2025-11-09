"""
Variable Mapping for Han 2012 Cox Model - TCGA STAD Edition
Maps TCGA STAD data to Han 2012 nomogram format
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class Han2012VariableMapper:
    """Maps TCGA STAD clinical data to Han 2012 nomogram format."""
    
    # T stage to depth of invasion mapping
    T_STAGE_MAPPING = {
        'T1': 'submucosa',
        'T1a': 'mucosa',
        'T1b': 'submucosa',
        'T2': 'proper_muscle',
        'T3': 'subserosa',
        'T4': 'serosa',
        'T4a': 'serosa',
        'T4b': 'adjacent_organ_invasion',
    }
    
    # N stage to estimated positive lymph nodes (midpoint of range)
    N_STAGE_TO_POSITIVE_LN = {
        'N0': 0,
        'N1': 2,    # Range 1-2, use 2
        'N2': 5,    # Range 3-6, use midpoint
        'N3': 11,   # Range 7-15, use midpoint
    }
    
    # Estimated examined lymph nodes by N stage (based on Han 2012 cohort)
    N_STAGE_TO_EXAMINED_LN = {
        'N0': 25,
        'N1': 28,
        'N2': 32,
        'N3': 35,
    }
    
    @staticmethod
    def map_age_category(age: Optional[float]) -> str:
        """Map continuous age to Han 2012 categories."""
        if pd.isna(age) or age is None:
            return '50-59'
        
        if age < 40:
            return '< 40'
        elif age < 50:
            return '40-49'
        elif age < 60:
            return '50-59'
        elif age < 70:
            return '60-69'
        else:
            return '>= 70'
    
    @staticmethod
    def map_sex(sex: Optional[str]) -> str:
        """Map sex to Han 2012 format."""
        if pd.isna(sex) or sex is None:
            return 'male'
        
        sex_lower = str(sex).lower().strip()
        if 'female' in sex_lower or sex_lower == 'f':
            return 'female'
        return 'male'
    
    @staticmethod
    def map_tumor_location(icd_code: Optional[str] = None) -> str:
        """
        Estimate tumor location (upper/middle/lower third) from ICD-10 code.
        
        If ICD-10 code is available, infer from site:
        - C16.0 (cardia): upper
        - C16.1 (fundus): upper
        - C16.2 (body): middle
        - C16.3 (pyloric antrum): lower
        - C16.4 (pylorus): lower
        - C16.5 (lesser curvature): middle
        - C16.6 (greater curvature): middle
        - C16.8 (overlapping): random based on distribution
        - C16.9 (NOS): random based on distribution
        
        If no ICD code, use epidemiological distribution: 
        Lower (60%), Middle (25%), Upper (15%).
        """
        if icd_code and not pd.isna(icd_code):
            icd_str = str(icd_code).strip().upper()
            if icd_str in ['C16.0', 'C16.1']:
                return 'upper'
            elif icd_str == 'C16.2':
                return 'middle'
            elif icd_str in ['C16.3', 'C16.4']:
                return 'lower'
            elif icd_str in ['C16.5', 'C16.6']:
                return 'middle'
            # For C16.8, C16.9, or others, use distribution
        
        # Default to distribution-based sampling
        import random
        locations = ['lower', 'middle', 'upper']
        weights = [0.60, 0.25, 0.15]  # Epidemiological distribution
        return random.choices(locations, weights=weights, k=1)[0]
    
    @staticmethod
    def map_depth_of_invasion(t_stage: Optional[str]) -> str:
        """Map T stage to depth of invasion category."""
        if pd.isna(t_stage) or t_stage is None:
            return 'proper_muscle'
        
        t_stage_clean = str(t_stage).strip()
        return Han2012VariableMapper.T_STAGE_MAPPING.get(
            t_stage_clean, 
            'proper_muscle'
        )
    
    @staticmethod
    def map_metastatic_lymph_nodes(n_stage: Optional[str], positive_ln: Optional[int] = None) -> str:
        """
        Map to Han 2012 positive lymph node categories.
        
        Args:
            n_stage: N stage (N0-N3)
            positive_ln: Actual positive count if available
        """
        if positive_ln is not None and not pd.isna(positive_ln):
            n_pos = int(positive_ln)
            if n_pos == 0:
                return '0'
            elif n_pos <= 2:
                return '1-2'
            elif n_pos <= 6:
                return '3-6'
            elif n_pos <= 15:
                return '7-15'
            else:
                return '>= 16'
        
        # Estimate from N stage
        if pd.isna(n_stage) or n_stage is None:
            return '1-2'
        
        n_stage_clean = str(n_stage).strip()
        
        # Map N stage to category
        if 'N0' in n_stage_clean:
            return '0'
        elif 'N1' in n_stage_clean:
            return '1-2'
        elif 'N2' in n_stage_clean:
            return '3-6'
        elif 'N3' in n_stage_clean:
            return '7-15'
        else:
            return '1-2'
    
    @staticmethod
    def estimate_examined_lymph_nodes(
        n_stage: Optional[str],
        examined_nodes: Optional[int] = None,
        positive_nodes: Optional[int] = None
    ) -> int:
        """
        Estimate number of examined lymph nodes.
        
        Priority:
        1. Actual count if available
        2. Estimated from N stage (based on Han 2012 cohort mean of 32.4)
        """
        # If we have actual count, use it
        if examined_nodes is not None and not pd.isna(examined_nodes):
            return max(int(examined_nodes), 15)
        
        # If we have positive nodes, examined must be at least that many + buffer
        if positive_nodes is not None and not pd.isna(positive_nodes):
            return max(int(positive_nodes) + 15, 15)
        
        # Estimate from N stage using Han 2012 cohort statistics
        if pd.isna(n_stage) or n_stage is None:
            return 30  # Default to cohort median
        
        n_stage_clean = str(n_stage).strip()
        return Han2012VariableMapper.N_STAGE_TO_EXAMINED_LN.get(
            n_stage_clean,
            30
        )
    
    @staticmethod
    def map_patient_from_dict(patient_dict: Dict) -> Dict:
        """
        Map a patient dictionary (from your risk calculator) to Han 2012 format.
        
        Args:
            patient_dict: Dictionary with keys like:
                - age, T_stage, N_stage, etc.
        
        Returns:
            Dictionary with Han 2012 variables
        """
        age = patient_dict.get('age')
        sex = patient_dict.get('sex', patient_dict.get('Sex'))
        t_stage = patient_dict.get('T_stage')
        n_stage = patient_dict.get('N_stage')
        positive_ln = patient_dict.get('positive_LN')
        examined_ln = patient_dict.get('total_LN')
        icd_10 = patient_dict.get('icd_10')
        
        han_patient = {
            'age': Han2012VariableMapper.map_age_category(age),
            'sex': Han2012VariableMapper.map_sex(sex),
            'location': Han2012VariableMapper.map_tumor_location(icd_10),
            'depth_of_invasion': Han2012VariableMapper.map_depth_of_invasion(t_stage),
            'metastatic_lymph_nodes': Han2012VariableMapper.map_metastatic_lymph_nodes(
                n_stage, positive_ln
            ),
            'examined_lymph_nodes': Han2012VariableMapper.estimate_examined_lymph_nodes(
                n_stage, examined_ln, positive_ln
            )
        }
        
        return han_patient
    
    @staticmethod
    def get_imputation_flags(patient_dict: Dict) -> Dict:
        """
        Track which variables were imputed vs directly measured.
        
        Returns:
            Dictionary of boolean flags
        """
        sex = patient_dict.get('sex', patient_dict.get('Sex'))
        positive_ln = patient_dict.get('positive_LN')
        examined_ln = patient_dict.get('total_LN')
        
        return {
            'age_available': patient_dict.get('age') is not None,
            'sex_available': sex is not None and not pd.isna(sex),
            'location_imputed': True,  # Always imputed for TCGA data
            'positive_ln_imputed': positive_ln is None or pd.isna(positive_ln),
            'examined_ln_imputed': examined_ln is None or pd.isna(examined_ln),
        }


def test_mapper():
    """Test the mapper with sample data."""
    print("Testing Han 2012 Variable Mapper for TCGA Data")
    print("=" * 60)
    
    test_patient = {
        'age': 65,
        'Sex': 'Female',
        'T_stage': 'T3',
        'N_stage': 'N2',
        'positive_LN': None,  # Not available in TCGA
        'total_LN': None,     # Not available in TCGA
    }
    
    mapper = Han2012VariableMapper()
    han_patient = mapper.map_patient_from_dict(test_patient)
    imputation_flags = mapper.get_imputation_flags(test_patient)
    
    print("\nOriginal Patient Data:")
    for key, value in test_patient.items():
        print(f"  {key}: {value}")
    
    print("\nMapped to Han 2012 Format:")
    for key, value in han_patient.items():
        print(f"  {key}: {value}")
    
    print("\nImputation Flags:")
    for key, value in imputation_flags.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_mapper()
