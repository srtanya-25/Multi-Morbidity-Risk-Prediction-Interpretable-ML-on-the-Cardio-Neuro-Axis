import pandas as pd
import numpy as np

# Load the engineered dataset
df = pd.read_csv('02_Processed_Data/FINAL_ENGINEERED_DATASET.csv')

print("="*80)
print("DATA LEAKAGE DIAGNOSTIC REPORT")
print("="*80)

# Check 1: Identify which columns are features vs targets
TARGET_COLS = [col for col in df.columns if col.startswith('TARGET_')]
FEATURE_COLS = [col for col in df.columns if not col.startswith('TARGET_')]

print(f"\nTotal Columns: {len(df.columns)}")
print(f"Target Columns: {len(TARGET_COLS)}")
print(f"Feature Columns: {len(FEATURE_COLS)}")

# Check 2: Find perfect correlations between features and targets
print("\n" + "="*80)
print("CHECKING FOR PERFECT CORRELATIONS (Feature → Target)")
print("="*80)

leakage_found = False
for target in TARGET_COLS:
    for feature in FEATURE_COLS:
        try:
            corr = df[feature].corr(df[target])
            if abs(corr) > 0.99:  # Near-perfect correlation
                print(f"\n⚠ LEAKAGE DETECTED!")
                print(f"   Feature: {feature}")
                print(f"   Target: {target}")
                print(f"   Correlation: {corr:.4f}")
                
                # Show sample values
                sample = df[[feature, target]].head(10)
                print(f"\n   Sample values:")
                print(sample.to_string(index=False))
                leakage_found = True
        except:
            pass

if not leakage_found:
    print("\n✓ No perfect correlations found")

# Check 3: Identify which original BRFSS columns map to targets
print("\n" + "="*80)
print("ORIGINAL BRFSS COLUMNS VS TARGET LABELS")
print("="*80)

mappings = {
    'DIABETE4': 'TARGET_T2DM',
    'CHCKDNY2': 'TARGET_CKD',
    'ADDEPEV3': 'TARGET_MDD',
    'BPHIGH6': 'TARGET_HTN',
    'CVDSTRK3': 'TARGET_STROKE',
    'CVDINFR4': 'TARGET_CAD'
}

for brfss_col, target_col in mappings.items():
    if brfss_col in df.columns and target_col in df.columns:
        # Check if they're identical
        match_rate = (df[brfss_col] == df[target_col]).mean()
        print(f"\n{brfss_col} → {target_col}")
        print(f"   Match rate: {match_rate:.2%}")
        if match_rate > 0.8:
            print(f"   ⚠ These are nearly identical! This IS the leakage source.")

# Check 4: Show which features should be REMOVED
print("\n" + "="*80)
print("RECOMMENDED FEATURES TO REMOVE")
print("="*80)

features_to_remove = []
for brfss_col in mappings.keys():
    if brfss_col in FEATURE_COLS:
        features_to_remove.append(brfss_col)
        print(f"   - {brfss_col} (directly answers the target question)")

# Additional problematic features
additional_risky = ['GENHLTH', 'MENTHLTH']
for col in additional_risky:
    if col in FEATURE_COLS:
        print(f"   - {col} (highly predictive of health outcomes)")

print("\n" + "="*80)
print("SOLUTION:")
print("="*80)
print("Remove these columns from FEATURE_COLS in model_training.py:")
print(f"   EXCLUDE_COLS = {features_to_remove + additional_risky}")
print("\nOr use ONLY the lab-derived features (AVG_A1C, AVG_HDL, etc.)")
print("="*80)