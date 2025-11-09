#!/usr/bin/env python3
"""
Gastric Cancer Staging Visualization Script

This script processes TCGA STAD PanCanAtlas 2018 clinical data to generate
visualizations of TNM staging distributions and survival outcomes.

Generated figures:
- tnm_staging_distribution.png: Bar plots for T, N, M, and overall stages
- tn_heatmap.png: Heatmap of T×N stage combinations
- os_by_stage.png: Overall survival by stage with median survival and event rates
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def validate_columns(df, required_columns):
    """Validate that required columns are present in the dataframe."""
    missing = []
    for col in required_columns:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def load_and_validate_data(data_path):
    """Load data and validate schema."""
    # Required column mappings
    column_mappings = {
        'Patient ID': 'patient_id',
        'Neoplasm Disease Stage American Joint Committee on Cancer Code': 'ajcc_stage',
        'American Joint Committee on Cancer Tumor Stage Code': 't_stage',
        'Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code': 'n_stage',
        'American Joint Committee on Cancer Metastasis Stage Code': 'm_stage',
        'Overall Survival (Months)': 'overall_survival_months',
        'Overall Survival Status': 'overall_survival_status'
    }

    # Load data
    df = pd.read_csv(data_path, sep='\t')

    # Validate required columns
    validate_columns(df, column_mappings.keys())

    # Rename columns
    df = df.rename(columns=column_mappings)

    # Convert survival months to numeric
    df['overall_survival_months'] = pd.to_numeric(df['overall_survival_months'], errors='coerce')

    # Convert survival status to binary (1=deceased, 0=living)
    df['deceased'] = df['overall_survival_status'].str.contains('DECEASED|1:', case=False, na=False).astype(int)

    return df

def harmonize_stages(df):
    """Standardize and harmonize staging labels."""
    # Clean T stages (remove sub-classifications like T2B -> T2)
    df['t_stage_clean'] = df['t_stage'].str.extract(r'(T[0-4])', expand=False)

    # Clean N stages
    df['n_stage_clean'] = df['n_stage'].str.extract(r'(N[0-3X])', expand=False)

    # Clean M stages
    df['m_stage_clean'] = df['m_stage'].str.extract(r'(M[0-1X])', expand=False)

    # Clean overall stage (extract roman numerals)
    df['stage_clean'] = df['ajcc_stage'].str.extract(r'STAGE\s+(I{1,4})', expand=False)

    # Map roman to arabic for sorting
    roman_to_arabic = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    df['stage_numeric'] = df['stage_clean'].map(roman_to_arabic)

    return df

def create_tnm_distribution_plot(df, output_path):
    """Create TNM staging distribution bar plots."""
    # Roman to arabic mapping for sorting
    roman_to_arabic = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('TNM Staging Distribution - TCGA STAD Cohort', fontsize=16, fontweight='bold')

    # T stage distribution
    t_counts = df['t_stage_clean'].value_counts().sort_index()
    axes[0,0].bar(t_counts.index, t_counts.values, color='skyblue', alpha=0.8)
    axes[0,0].set_title('T Stage Distribution')
    axes[0,0].set_xlabel('T Stage')
    axes[0,0].set_ylabel('Number of Patients')
    axes[0,0].grid(True, alpha=0.3)

    # N stage distribution
    n_counts = df['n_stage_clean'].value_counts().sort_index()
    axes[0,1].bar(n_counts.index, n_counts.values, color='lightgreen', alpha=0.8)
    axes[0,1].set_title('N Stage Distribution')
    axes[0,1].set_xlabel('N Stage')
    axes[0,1].set_ylabel('Number of Patients')
    axes[0,1].grid(True, alpha=0.3)

    # M stage distribution
    m_counts = df['m_stage_clean'].value_counts().sort_index()
    axes[1,0].bar(m_counts.index, m_counts.values, color='salmon', alpha=0.8)
    axes[1,0].set_title('M Stage Distribution')
    axes[1,0].set_xlabel('M Stage')
    axes[1,0].set_ylabel('Number of Patients')
    axes[1,0].grid(True, alpha=0.3)

    # Overall stage distribution
    stage_counts = df['stage_clean'].value_counts().sort_index(key=lambda x: [roman_to_arabic.get(s, 0) for s in x])
    axes[1,1].bar(stage_counts.index, stage_counts.values, color='purple', alpha=0.8)
    axes[1,1].set_title('Overall Stage Distribution')
    axes[1,1].set_xlabel('Stage')
    axes[1,1].set_ylabel('Number of Patients')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_tn_heatmap(df, output_path):
    """Create T×N stage combination heatmap."""
    # Create crosstab of T and N stages
    tn_crosstab = pd.crosstab(df['t_stage_clean'], df['n_stage_clean'])

    # Sort indices
    t_order = sorted(tn_crosstab.index, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    n_order = sorted(tn_crosstab.columns, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)

    tn_crosstab = tn_crosstab.reindex(index=t_order, columns=n_order)

    plt.figure(figsize=(10, 8))
    sns.heatmap(tn_crosstab, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Number of Patients'})
    plt.title('T×N Stage Combinations - TCGA STAD Cohort', fontsize=14, fontweight='bold')
    plt.xlabel('N Stage')
    plt.ylabel('T Stage')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_survival_by_stage_plot(df, output_path):
    """Create overall survival by stage plot."""
    # Calculate median survival and event rates by stage
    survival_stats = df.groupby('stage_clean').agg({
        'overall_survival_months': ['median', 'count'],
        'deceased': 'mean'
    }).round(2)

    survival_stats.columns = ['median_survival_months', 'n_patients', 'event_rate']
    survival_stats = survival_stats.reset_index()

    # Sort by stage
    roman_to_arabic = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    survival_stats['stage_order'] = survival_stats['stage_clean'].map(roman_to_arabic)
    survival_stats = survival_stats.sort_values('stage_order')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Overall Survival by Stage - TCGA STAD Cohort', fontsize=14, fontweight='bold')

    # Median survival plot
    bars1 = ax1.bar(survival_stats['stage_clean'], survival_stats['median_survival_months'],
                    color='steelblue', alpha=0.8)
    ax1.set_title('Median Overall Survival')
    ax1.set_xlabel('Stage')
    ax1.set_ylabel('Median Survival (Months)')
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars1, survival_stats['median_survival_months']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

    # Event rate plot
    bars2 = ax2.bar(survival_stats['stage_clean'], survival_stats['event_rate'] * 100,
                    color='darkred', alpha=0.8)
    ax2.set_title('Event Rate (Mortality)')
    ax2.set_xlabel('Stage')
    ax2.set_ylabel('Event Rate (%)')
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars2, survival_stats['event_rate']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value*100:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate gastric cancer staging visualizations from TCGA data')
    parser.add_argument('--data', default='data/tcga_2018_clinical_data.tsv',
                       help='Path to TCGA clinical data TSV file')
    parser.add_argument('--output-dir', default='.',
                       help='Directory to save output figures')

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and validate data
    print("Loading and validating data...")
    df = load_and_validate_data(args.data)
    print(f"Loaded {len(df)} patient records")

    # Harmonize stages
    print("Harmonizing staging labels...")
    df = harmonize_stages(df)

    # Generate figures
    print("Generating TNM distribution plot...")
    create_tnm_distribution_plot(df, output_dir / 'tnm_staging_distribution.png')

    print("Generating T×N heatmap...")
    create_tn_heatmap(df, output_dir / 'tn_heatmap.png')

    print("Generating survival by stage plot...")
    create_survival_by_stage_plot(df, output_dir / 'os_by_stage.png')

    print(f"Figures saved to {output_dir.absolute()}")

if __name__ == '__main__':
    main()