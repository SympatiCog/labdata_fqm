#!/usr/bin/env python3
"""
Synthetic Data Generator for Laboratory Research Data

This script generates realistic synthetic data for the lab data browser application.
It creates CSV files with specified variables maintaining realistic correlations.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def generate_custom_ids(n_participants):
    """Generate realistic customID values following the M########format"""
    ids = []
    for i in range(n_participants):
        # Generate 8-digit number starting with 109 (based on real data pattern)
        number = np.random.randint(10900000, 11000000)
        ids.append(f"ID{number}")
    return ids

def generate_demographics(custom_ids):
    """Generate demographics data with realistic age and sex distributions"""
    n = len(custom_ids)
    
    # Age distribution: broader range from 10-75 based on real data
    ages = np.random.choice(
        range(10, 76), 
        size=n, 
        p=np.ones(66) / 66  # Uniform distribution for simplicity
    )
    
    # Sex: 1.0=Female, 2.0=Male (roughly 50/50 split)
    sex = np.random.choice([1.0, 2.0], size=n, p=[0.5, 0.5])
    
    return pd.DataFrame({
        'customID': custom_ids,
        'age': ages,
        'sex': sex
    })

def generate_vo2max_data(custom_ids, ages):
    """Generate VO2max and related physiological data with age correlations"""
    n = len(custom_ids)
    
    # VO2max decreases with age, add noise
    base_vo2 = 50 - (ages - 20) * 0.3
    vo2_max = np.maximum(4.0, base_vo2 + np.random.normal(0, 8, n))
    vo2_max = np.minimum(vo2_max, 45.0)  # Cap at reasonable max
    
    # HR_rest: typically 40-100, slight increase with age
    HR_rest = 60 + (ages - 30) * 0.2 + np.random.normal(0, 15, n)
    HR_rest = np.clip(HR_rest, 40, 120)
    
    # HR_max: decreases with age (220 - age formula with variation)
    HR_max = 220 - ages + np.random.normal(0, 10, n)
    HR_max = np.clip(HR_max, 120, 200)
    
    # RER values: typically 0.7-1.3
    RER_rest = np.random.normal(0.8, 0.1, n)
    RER_rest = np.clip(RER_rest, 0.65, 1.1)
    
    RER_max = np.random.normal(1.05, 0.15, n)
    RER_max = np.clip(RER_max, 0.8, 1.35)
    
    return pd.DataFrame({
        'customID': custom_ids,
        'vo2_max': np.round(vo2_max, 1),
        'HR_rest': np.round(HR_rest, 0).astype(int),
        'HR_max': np.round(HR_max, 0).astype(int),
        'RER_rest': np.round(RER_rest, 2),
        'RER_max': np.round(RER_max, 2)
    })

def generate_additional_demos(custom_ids):
    """Generate additional demographic variables"""
    n = len(custom_ids)
    
    # Race distribution (based on observed values 1.0-6.0, with 5.0 being most common)
    race_probs = [0.05, 0.10, 0.05, 0.10, 0.60, 0.10]  # 5.0 is most common
    race = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], size=n, p=race_probs)
    
    # Ethnicity (1.0-3.0, with 2.0 being most common)
    ethnicity = np.random.choice([1.0, 2.0, 3.0], size=n, p=[0.1, 0.8, 0.1])
    
    # Native English speaker (1.0=No, 2.0=Yes, mostly Yes)
    native_language_english = np.random.choice([1.0, 2.0], size=n, p=[0.15, 0.85])
    
    return pd.DataFrame({
        'customID': custom_ids,
        'race': race,
        'ethnicity': ethnicity,
        'native_language_english': native_language_english
    })

def generate_flanker_data(custom_ids, ages):
    """Generate flanker task data with age-related performance effects"""
    n = len(custom_ids)
    
    # Practice accuracy: generally high, slight decrease with age
    practice_acc = 0.95 - (ages - 20) * 0.001 + np.random.normal(0, 0.05, n)
    practice_acc = np.clip(practice_acc, 0.7, 1.0)
    
    # Reaction times: increase with age, congruent < neutral < incongruent
    base_rt = 400 + (ages - 20) * 3  # Base RT increases with age
    
    rt_con = base_rt + np.random.normal(0, 50, n)  # Congruent fastest
    rt_neut = rt_con + np.random.normal(30, 20, n)  # Neutral intermediate
    rt_inc = rt_neut + np.random.normal(40, 25, n)  # Incongruent slowest
    
    # Clip to reasonable ranges
    rt_con = np.clip(rt_con, 300, 800)
    rt_neut = np.clip(rt_neut, 320, 850)
    rt_inc = np.clip(rt_inc, 350, 900)
    
    # Accuracy: congruent > neutral > incongruent, decreases with age
    base_acc = 0.92 - (ages - 20) * 0.002
    
    acc_con = base_acc + np.random.normal(0.05, 0.03, n)
    acc_neut = base_acc + np.random.normal(0.0, 0.04, n)
    acc_inc = base_acc + np.random.normal(-0.08, 0.05, n)
    
    # Clip accuracy values
    acc_con = np.clip(acc_con, 0.7, 1.0)
    acc_neut = np.clip(acc_neut, 0.65, 1.0)
    acc_inc = np.clip(acc_inc, 0.6, 1.0)
    
    return pd.DataFrame({
        'customID': custom_ids,
        'practice_acc': np.round(practice_acc, 3),
        'rt_con': np.round(rt_con, 1),
        'rt_inc': np.round(rt_inc, 1),
        'rt_neut': np.round(rt_neut, 1),
        'acc_con': np.round(acc_con, 3),
        'acc_inc': np.round(acc_inc, 3),
        'acc_neut': np.round(acc_neut, 3)
    })

def generate_cognitive_data(custom_ids, ages):
    """Generate Woodcock-Johnson cognitive test scores with age effects"""
    n = len(custom_ids)
    
    # Cognitive scores: some decline with age, add individual variation
    base_score = 45 - (ages - 30) * 0.1  # Slight decline with age
    
    # Add correlations between cognitive measures
    cog1_total = base_score + np.random.normal(0, 8, n)
    cog2_total = cog1_total * 0.7 + np.random.normal(0, 6, n)  # Correlated with cog1
    cog3_total = (cog1_total + cog2_total) * 0.4 + np.random.normal(0, 4, n)  # Correlated with both
    
    # Clip to observed ranges
    cog1_total = np.clip(cog1_total, 21, 55)
    cog2_total = np.clip(cog2_total, 15, 42)
    cog3_total = np.clip(cog3_total, 14, 30)
    
    return pd.DataFrame({
        'customID': custom_ids,
        'cog1_total': np.round(cog1_total, 1),
        'cog2_total': np.round(cog2_total, 1),
        'cog3_total': np.round(cog3_total, 1)
    })

def main():
    """Generate all synthetic datasets"""
    # Configuration
    n_participants = 200  # Generate 200 synthetic participants
    output_dir = Path("synthetic_data")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print(f"Generating synthetic data for {n_participants} participants...")
    
    # Generate participant IDs
    custom_ids = generate_custom_ids(n_participants)
    
    # Generate demographics (this determines ages for other correlations)
    demographics = generate_demographics(custom_ids)
    ages = demographics['age'].values
    
    print("Generating demographics.csv...")
    demographics.to_csv(output_dir / "demographics.csv", index=False)
    
    print("Generating VO2max.csv...")
    vo2_data = generate_vo2max_data(custom_ids, ages)
    vo2_data.to_csv(output_dir / "VO2max.csv", index=False)
    
    print("Generating additional_demos.csv...")
    additional_demos = generate_additional_demos(custom_ids)
    additional_demos.to_csv(output_dir / "additional_demos.csv", index=False)
    
    print("Generating flanker.csv...")
    flanker_data = generate_flanker_data(custom_ids, ages)
    flanker_data.to_csv(output_dir / "flanker.csv", index=False)
    
    print("Generating woodcock_johnson.csv...")
    cognitive_data = generate_cognitive_data(custom_ids, ages)
    cognitive_data.to_csv(output_dir / "woodcock_johnson.csv", index=False)
    
    print(f"\nSynthetic data generation complete!")
    print(f"Files created in '{output_dir}' directory:")
    print("- demographics.csv")
    print("- VO2max.csv") 
    print("- additional_demos.csv")
    print("- flanker.csv")
    print("- woodcock_johnson.csv")
    
    # Print summary statistics
    print(f"\nDataset summary:")
    print(f"- Total participants: {n_participants}")
    print(f"- Age range: {demographics['age'].min()}-{demographics['age'].max()}")
    print(f"- Sex distribution: {(demographics['sex'] == 1.0).sum()} Female, {(demographics['sex'] == 2.0).sum()} Male")
    print(f"- VO2max range: {vo2_data['vo2_max'].min():.1f}-{vo2_data['vo2_max'].max():.1f}")

if __name__ == "__main__":
    main()
