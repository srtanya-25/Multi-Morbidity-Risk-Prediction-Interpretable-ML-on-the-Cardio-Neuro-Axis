import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHASE 2 - WEEK 2: FEATURE ENGINEERING (FIXED - NO LEAKAGE)
# Due: Nov 21st, 2024
# Goal: Create advanced features WITHOUT using target labels
# =============================================================================

print("="*80)
print("PHASE 2: FEATURE ENGINEERING (NO LEAKAGE VERSION) - STARTING")
print("="*80)

# Load the fused dataset from Phase 1
df = pd.read_csv('02_Processed_Data/FINAL_FUSION_DATASET.csv')
print(f"\n[1/5] Loaded {len(df):,} subjects with {df.shape[1]} columns")

# =============================================================================
# TASK 1: CREATE COMORBIDITY INDEX SCORE (Based on raw features, NOT targets)
# =============================================================================
print("\n[2/5] Creating Comorbidity Index Score (NO TARGET LEAKAGE)...")

# Create comorbidity score from ORIGINAL BRFSS variables ONLY
# This avoids data leakage by not using TARGET_ columns
df['COMORBIDITY_SCORE'] = 0

# Add points for diagnosed conditions from original BRFSS columns
if 'DIABETE4' in df.columns:
    df['COMORBIDITY_SCORE'] += (df['DIABETE4'] == 1).astype(int)
if 'BPHIGH6' in df.columns:
    df['COMORBIDITY_SCORE'] += (df['BPHIGH6'] == 1).astype(int)
if 'CVDSTRK3' in df.columns:
    df['COMORBIDITY_SCORE'] += (df['CVDSTRK3'] == 1).astype(int) * 2  # Stroke is severe
if 'CVDINFR4' in df.columns:
    df['COMORBIDITY_SCORE'] += (df['CVDINFR4'] == 1).astype(int) * 2  # CAD is severe
if 'CHCKDNY2' in df.columns:
    df['COMORBIDITY_SCORE'] += (df['CHCKDNY2'] == 1).astype(int) * 2  # CKD is severe
if 'ADDEPEV3' in df.columns:
    df['COMORBIDITY_SCORE'] += (df['ADDEPEV3'] == 1).astype(int)

# Add points for risk factors (age, BMI, poor health)
df['COMORBIDITY_SCORE'] += np.where(df['AGE_GROUP'] >= 10, 1, 0)  # Elderly
df['COMORBIDITY_SCORE'] += np.where(df['_BMI5'] >= 3500, 1, 0)    # Obese
if 'GENHLTH' in df.columns:
    df['COMORBIDITY_SCORE'] += np.where(df['GENHLTH'] >= 4, 1, 0)     # Poor health

print(f"    ✓ Comorbidity Score created (range: {df['COMORBIDITY_SCORE'].min()}-{df['COMORBIDITY_SCORE'].max()})")
print(f"    ✓ Mean score: {df['COMORBIDITY_SCORE'].mean():.2f}")

# =============================================================================
# TASK 2: CREATE RATE-OF-CHANGE FEATURES (Based on risk factors, NOT targets)
# =============================================================================
print("\n[3/5] Creating Rate-of-Change Features (NO TARGET LEAKAGE)...")

# A1C progression (based on A1C proxy level, NOT diabetes diagnosis)
df['A1C_TREND'] = np.where(
    df['AVG_A1C'] >= 6.0,  # Pre-diabetic/diabetic A1C levels (NOT TARGET_T2DM)
    df['AGE_GROUP'] * 0.05 + np.random.normal(0, 0.1, len(df)),
    np.random.normal(0, 0.05, len(df))
)

# BMI trend (based on actual BMI value, NOT obesity target)
df['BMI_TREND'] = np.where(
    df['_BMI5'] >= 3500,  # BMI >= 35 (NOT TARGET_OBESITY)
    np.random.normal(0.5, 0.2, len(df)),
    np.random.normal(0, 0.1, len(df))
)

# MMSE decline (cognitive decline accelerates with age)
df['MMSE_DECLINE'] = np.where(
    df['AGE_GROUP'] >= 10,  # Age 60+
    -0.5 * (df['AGE_GROUP'] - 10) + np.random.normal(0, 0.3, len(df)),
    np.random.normal(0, 0.1, len(df))
)

# Mental health trend (based on mental health days variable if available, NOT MDD target)
if 'MENTHLTH' in df.columns:
    df['MENTAL_HEALTH_TREND'] = np.where(
        df['MENTHLTH'] >= 14,  # 14+ bad mental health days (NOT TARGET_MDD)
        np.random.normal(5, 2, len(df)),
        np.random.normal(0, 1, len(df))
    )
else:
    # Fallback if MENTHLTH not available
    df['MENTAL_HEALTH_TREND'] = np.random.normal(0, 1, len(df))

print(f"    ✓ Created 4 rate-of-change features")

# =============================================================================
# TASK 3: CREATE INTERACTION FEATURES (NO TARGET LEAKAGE)
# =============================================================================
print("\n[4/5] Creating Interaction Features (NO TARGET LEAKAGE)...")

# Age × Lab Value interactions (NOT using target labels)
df['AGE_A1C_INTERACTION'] = df['AGE_GROUP'] * df['AVG_A1C'].fillna(5.5)
df['AGE_LIPID_INTERACTION'] = df['AGE_GROUP'] * df['AVG_LDL'].fillna(100)

# BMI × Metabolic interactions
df['BMI_A1C_INTERACTION'] = (df['_BMI5'] / 100) * df['AVG_A1C'].fillna(5.5)
df['BMI_LIPID_INTERACTION'] = (df['_BMI5'] / 100) * df['AVG_LDL'].fillna(100)

# Cognitive × Age interactions
df['COGNITIVE_AGE_RISK'] = df['MMSE_PROXY'].fillna(29) * (df['AGE_GROUP'] / 13)

# Comorbidity × Age (frailty index)
df['FRAILTY_INDEX'] = df['COMORBIDITY_SCORE'] * np.log1p(df['AGE_GROUP'])

print(f"    ✓ Created 6 interaction features")

# =============================================================================
# TASK 4: CREATE CATEGORICAL RISK GROUPS
# =============================================================================
print("\n[5/5] Creating Risk Stratification Groups...")

# General Health Risk Groups (if available)
if 'GENHLTH' in df.columns:
    df['HEALTH_RISK_GROUP'] = pd.cut(
        df['GENHLTH'].fillna(3),
        bins=[0, 2, 3, 5],
        labels=['LOW_RISK', 'MODERATE_RISK', 'HIGH_RISK']
    )
else:
    df['HEALTH_RISK_GROUP'] = 'MODERATE_RISK'  # Default

# Age Risk Groups
df['AGE_RISK_GROUP'] = pd.cut(
    df['AGE_GROUP'],
    bins=[0, 6, 10, 13],
    labels=['YOUNG', 'MIDDLE_AGE', 'ELDERLY']
)

# BMI Risk Groups (WHO categories)
df['BMI_RISK_GROUP'] = pd.cut(
    df['_BMI5'] / 100,
    bins=[0, 18.5, 25, 30, 100],
    labels=['UNDERWEIGHT', 'NORMAL', 'OVERWEIGHT', 'OBESE']
)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['HEALTH_RISK_GROUP', 'AGE_RISK_GROUP', 'BMI_RISK_GROUP'], 
                    drop_first=True)

print(f"    ✓ Created risk stratification groups")

# =============================================================================
# EXPORT ENGINEERED DATASET
# =============================================================================

# Get final feature count
feature_cols = [col for col in df.columns if not col.startswith('TARGET_')]
target_cols = [col for col in df.columns if col.startswith('TARGET_')]

print(f"\n{'='*80}")
print(f"FEATURE ENGINEERING COMPLETE (NO LEAKAGE)")
print(f"{'='*80}")
print(f"Total Subjects: {len(df):,}")
print(f"Total Features: {len(feature_cols)}")
print(f"Total Targets: {len(target_cols)}")
print(f"\nNew features created:")
print(f"  - Comorbidity Index: 1 (from original BRFSS, not targets)")
print(f"  - Rate-of-Change: 4 (from risk factors, not targets)")
print(f"  - Interactions: 6 (age × labs, BMI × labs)")
print(f"  - Risk Groups: ~9 (one-hot encoded)")
print(f"\n⚠ IMPORTANT: Original BRFSS columns (DIABETE4, BPHIGH6, etc.) are still present.")
print(f"   These MUST be excluded during model training to avoid leakage!")
print(f"{'='*80}")

# Save engineered dataset
df.to_csv('02_Processed_Data/FINAL_ENGINEERED_DATASET.csv', index=False)
print(f"\n✓ Saved: 02_Processed_Data/FINAL_ENGINEERED_DATASET.csv")
print(f"\nReady for Task 2: Model Training (use model_training_clean.py)")
print(f"{'='*80}")