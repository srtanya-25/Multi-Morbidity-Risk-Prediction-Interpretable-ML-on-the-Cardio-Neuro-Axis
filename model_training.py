import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, fallback to RandomForest if not available
try:
    import xgboost as xgb
    USE_XGBOOST = True
    print("Using XGBoost models")
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    USE_XGBOOST = False
    print("⚠ XGBoost not found. Using RandomForest")

# =============================================================================
# PHASE 2 - WEEK 2: MODEL TRAINING (NO LEAKAGE VERSION)
# Due: Nov 26th, 2024
# Goal: Train production-ready multi-label classifier WITHOUT data leakage
# =============================================================================

print("="*80)
print("PHASE 2: MULTI-LABEL MODEL TRAINING (NO LEAKAGE)")
print("="*80)

# Ensure output directory exists
Path('03_Models').mkdir(exist_ok=True)

# =============================================================================
# STEP 1: LOAD ENGINEERED DATASET
# =============================================================================
print("\n[1/6] Loading engineered dataset...")

df = pd.read_csv('02_Processed_Data/FINAL_ENGINEERED_DATASET.csv')
print(f"    ✓ Loaded {len(df):,} subjects")

# =============================================================================
# STEP 2: DEFINE FEATURES (EXCLUDE LEAKING COLUMNS)
# =============================================================================
print("\n[2/6] Defining features (excluding survey response columns)...")

# Define targets
TARGET_COLS = [col for col in df.columns if col.startswith('TARGET_')]

# CRITICAL: Explicitly exclude columns that directly answer target questions
EXCLUDE_COLS = [
    # Survey responses that directly answer targets (DATA LEAKAGE!)
    'DIABETE4',   # Directly answers TARGET_T2DM
    'CHCKDNY2',   # Directly answers TARGET_CKD
    'ADDEPEV3',   # Directly answers TARGET_MDD
    'BPHIGH6',    # Directly answers TARGET_HTN
    'CVDSTRK3',   # Directly answers TARGET_STROKE
    'CVDINFR4',   # Directly answers TARGET_CAD
    'GENHLTH',    # Too directly predictive of health outcomes
    'MENTHLTH',   # Too directly predictive of mental health
    # Administrative columns
    '_STATE',
    '_LLCPWT'
]

# Get all feature columns except targets and excluded ones
FEATURE_COLS = [col for col in df.columns 
                if col not in TARGET_COLS 
                and col not in EXCLUDE_COLS
                and not col.startswith('TARGET_')]

print(f"    ✓ Total columns: {len(df.columns)}")
print(f"    ✓ Target columns: {len(TARGET_COLS)}")
print(f"    ✓ Excluded columns: {len(EXCLUDE_COLS)}")
print(f"    ✓ Feature columns: {len(FEATURE_COLS)}")

# Verify no leakage
leaking_keywords = ['DIABETE', 'BPHIGH', 'CVDSTRK', 'CVDINFR', 'CHCKDNY', 'ADDEPEV']
leaked = [col for col in FEATURE_COLS if any(kw in col.upper() for kw in leaking_keywords)]
if leaked:
    print(f"    ⚠ ERROR: Found leaking columns in features: {leaked}")
    exit(1)

print(f"    ✓ Leakage check passed!")
print(f"    ✓ Using: Lab values, demographics, BMI, MMSE, engineered features")

X = df[FEATURE_COLS].copy()
y = df[TARGET_COLS].copy()

# =============================================================================
# STEP 3: PREPROCESSING
# =============================================================================
print("\n[3/6] Preprocessing features...")

# Impute missing values
print(f"    → Imputing missing values...")
X = X.fillna(X.median(numeric_only=True))

# Standardize features
print(f"    → Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=FEATURE_COLS)

print(f"    ✓ Preprocessing complete")

# =============================================================================
# STEP 4: TRAIN-TEST SPLIT (Stratified)
# =============================================================================
print("\n[4/6] Creating train-test split...")

# Use first target for stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y[TARGET_COLS[0]]  # Stratify on T2DM
)

print(f"    ✓ Train set: {len(X_train):,} samples")
print(f"    ✓ Test set: {len(X_test):,} samples")

# =============================================================================
# STEP 5: TRAIN MULTI-LABEL MODELS
# =============================================================================
print("\n[5/6] Training models for each target...")

models = {}
performance = {}

for i, target in enumerate(TARGET_COLS, 1):
    print(f"\n    [{i}/{len(TARGET_COLS)}] Training: {target}")
    
    # Check class balance
    pos_rate = y_train[target].mean()
    print(f"        Positive rate: {pos_rate:.2%}")
    
    # Skip targets with 0% or 100% positive rate (can't train)
    if pos_rate == 0 or pos_rate == 1:
        print(f"        ⚠ SKIPPED: Cannot train (only one class present)")
        models[target] = None
        performance[target] = {
            'AUC': 0.5,
            'AUPRC': pos_rate,
            'Accuracy': pos_rate if pos_rate == 1 else (1 - pos_rate),
            'Positive_Rate': pos_rate,
            'Status': 'SKIPPED'
        }
        continue
    
    # Handle class imbalance
    scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1
    
    # Train model (XGBoost or RandomForest)
    if USE_XGBOOST:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train, 
            y_train[target],
            eval_set=[(X_test, y_test[target])],
            verbose=False
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train[target])
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics (handle edge cases)
    try:
        auc = roc_auc_score(y_test[target], y_pred_proba)
        auprc = average_precision_score(y_test[target], y_pred_proba)
    except ValueError:
        auc, auprc = 0.5, pos_rate  # Fallback if only one class in test set
    
    accuracy = (y_pred == y_test[target]).mean()
    
    performance[target] = {
        'AUC': auc,
        'AUPRC': auprc,
        'Accuracy': accuracy,
        'Positive_Rate': pos_rate,
        'Status': 'TRAINED'
    }
    
    print(f"        AUC: {auc:.3f} | AUPRC: {auprc:.3f} | Acc: {accuracy:.3f}")
    
    # Save model
    models[target] = model

# =============================================================================
# STEP 6: SAVE MODELS AND ARTIFACTS
# =============================================================================
print("\n[6/6] Saving models and preprocessing artifacts...")

# Save all models
joblib.dump(models, '03_Models/multi_label_xgboost_models.pkl')
print(f"    ✓ Saved models: 03_Models/multi_label_xgboost_models.pkl")

# Save scaler
joblib.dump(scaler, '03_Models/feature_scaler.pkl')
print(f"    ✓ Saved scaler: 03_Models/feature_scaler.pkl")

# Save feature names
with open('03_Models/feature_names.json', 'w') as f:
    json.dump(FEATURE_COLS, f)
print(f"    ✓ Saved feature names: 03_Models/feature_names.json")

# Save target names
with open('03_Models/target_names.json', 'w') as f:
    json.dump(TARGET_COLS, f)
print(f"    ✓ Saved target names: 03_Models/target_names.json")

# =============================================================================
# STEP 7: GENERATE PERFORMANCE REPORT
# =============================================================================
print("\n[7/7] Generating performance report...")

# Convert to DataFrame
perf_df = pd.DataFrame(performance).T
perf_df = perf_df.sort_values('AUC', ascending=False)

print(f"\n{'='*80}")
print(f"MODEL TRAINING COMPLETE - PERFORMANCE SUMMARY")
print(f"{'='*80}")
print(perf_df.to_string())
print(f"\n{'='*80}")

# Save performance metrics
perf_df.to_csv('03_Models/model_performance.csv')
print(f"\n✓ Performance report saved: 03_Models/model_performance.csv")

# Calculate overall metrics (only for trained models)
trained_models = perf_df[perf_df['Status'] == 'TRAINED']
if len(trained_models) > 0:
    mean_auc = trained_models['AUC'].mean()
    mean_auprc = trained_models['AUPRC'].mean()
    
    print(f"\nOVERALL PERFORMANCE (Trained Models Only):")
    print(f"  Mean AUC: {mean_auc:.3f}")
    print(f"  Mean AUPRC: {mean_auprc:.3f}")
    print(f"  Number of trained models: {len(trained_models)}/{len(TARGET_COLS)}")
    
    if mean_auc >= 0.75:
        print(f"\n✓ PASS: Models meet performance threshold (AUC ≥ 0.75)")
    else:
        print(f"\n⚠ WARNING: Models below threshold. Consider hyperparameter tuning.")
else:
    print(f"\n⚠ WARNING: No models were successfully trained!")

print(f"\n{'='*80}")
print(f"Ready for Week 3: SHAP Integration & API Development")
print(f"{'='*80}")