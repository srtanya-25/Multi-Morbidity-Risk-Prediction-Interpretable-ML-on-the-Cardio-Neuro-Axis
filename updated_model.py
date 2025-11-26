"""
Multimorbidity Multi-Label Classification: Train & compare models for ALL 12 diseases

Models included:
1. Logistic Regression
2. Random Forest
3. Neural Network (MLP)
4. XGBoost
5. LightGBM
6. CatBoost

How it works:
- Loads dataset from specified CSV path
- Identifies ALL target columns (TARGET_* or disease-specific columns)
- Excludes leaking columns that directly answer target questions
- Trains SEPARATE models for EACH disease (multi-label approach)
- Reports comprehensive metrics for each disease
- Saves all trained models

Requirements:
scikit-learn, pandas, numpy, xgboost, lightgbm, catboost, joblib

Run:
python multimorbidity_model_compare.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, average_precision_score, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# optional libraries
try:
    import xgboost as xgb
    USE_XGBOOST = True
except Exception:
    xgb = None
    USE_XGBOOST = False
    
try:
    import lightgbm as lgb
except Exception:
    lgb = None
    
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

import joblib
import json

# ------------------------- Helper functions -------------------------

def detect_target_columns(df):
    """
    Detect all target columns (diseases to predict)
    Looks for columns starting with TARGET_ or common disease names
    """
    # Primary method: columns starting with TARGET_
    target_cols = [col for col in df.columns if col.startswith('TARGET_')]
    
    if len(target_cols) == 0:
        # Fallback: look for disease-specific columns
        disease_keywords = ['T2DM', 'CKD', 'MDD', 'HTN', 'STROKE', 'CAD', 
                           'DIABETES', 'KIDNEY', 'DEPRESSION', 'HYPERTENSION',
                           'OBESE', 'OBESITY']
        target_cols = [col for col in df.columns 
                      if any(kw in col.upper() for kw in disease_keywords)
                      and not col.startswith('_')]
    
    return target_cols


def get_leaking_columns():
    """
    Define columns that directly answer target questions (DATA LEAKAGE!)
    """
    EXCLUDE_COLS = [
        'ADDEPEV3',
        'AGE_A1C_INTERACTION',
        'AGE_GROUP',
        'AGE_RISK_GROUP_ELDERLY',
        'AGE_RISK_GROUP_MIDDLE_AGE',
        'AGE_LIPID_INTERACTION',
        'BLOODCHO',
        'BMI',
        'BMI_A1C_INTERACTION',
        'BMI_CATEGORY',
        'BMI_LIPID_INTERACTION',
        'BMI_RISK_GROUP_NORMAL',
        'BMI_RISK_GROUP_OBESE',
        'BMI_RISK_GROUP_OVERWEIGHT',
        'BMI_RISK_GROUP_UNDERWEIGHT',
        'BMI_TREND',
        'BPHIGH6',
        'CHCCOPD3',
        'CHCKDNY',
        'CHCKDNY2',
        'CHCOCNC',
        'CHCSCNC',
        'COGNITIVE_AGE_RISK',
        'COMORBIDITY_SCORE',
        'CVDCRHD4',
        'CVDINFR4',
        'CVDSTRK3',
        'DIABETE4',
        'FRAILTY_INDEX',
        'GENHLTH',
        'HAVARTH4',
        'HEALTH_RISK_GROUP_HIGH_RISK',
        'HEALTH_RISK_GROUP_MODERATE_RISK',
        'HTM4',
        'MENTAL_HEALTH_TREND',
        'MENTHLTH',
        'MMSE',
        'MMSE_DECLINE',
        'MMSE_PROXY',
        'MMSE_SCORE',
        'MMSE_STD',
        'PHYSHLTH',
        'TOLDHI2',
        'TOLDHI3',
        'WTKG3',
        '_BMI5',
        '_BMI5CAT',
        '_CHOLCHK',
        '_CHOLCHK3',
        '_LLCPWT',
        '_RFBMI5',
        '_STATE',
        '_TREND',
        '_INTERACTION'
    ]
    return EXCLUDE_COLS


def train_and_eval_single_target(model, X_train, X_test, y_train, y_test, model_name, target_name, save_dir='models'):
    """Train model for a single target/disease and evaluate"""
    
    # Check class balance
    pos_rate = y_train.mean()
    
    # Skip if only one class present
    if pos_rate == 0 or pos_rate == 1:
        print(f"      ⚠ SKIPPED: Only one class present (pos_rate={pos_rate:.2%})")
        return dict(
            model_name=model_name,
            target=target_name,
            accuracy=pos_rate if pos_rate == 1 else (1 - pos_rate),
            precision=0, 
            recall=0, 
            f1=0, 
            roc_auc=0.5,
            auprc=pos_rate,
            pos_rate=pos_rate,
            status='SKIPPED'
        )
    
    # Train model
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return dict(
            model_name=model_name,
            target=target_name,
            accuracy=0, precision=0, recall=0, f1=0,
            roc_auc=0.5, auprc=pos_rate, pos_rate=pos_rate,
            status='FAILED'
        )
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Get probability scores
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        try:
            y_prob = model.decision_function(X_test)
        except Exception:
            y_prob = None
    else:
        y_prob = None

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate AUC and AUPRC (handle edge cases)
    try:
        auc = roc_auc_score(y_test, y_prob) if (y_prob is not None and len(np.unique(y_test))==2) else None
        auprc = average_precision_score(y_test, y_prob) if (y_prob is not None and len(np.unique(y_test))==2) else None
    except ValueError:
        test_pos_rate = y_test.mean()
        auc, auprc = 0.5, test_pos_rate

    print(f"      Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

    return dict(
        model_name=model_name,
        target=target_name,
        accuracy=acc, 
        precision=prec, 
        recall=rec, 
        f1=f1, 
        roc_auc=auc if auc is not None else 0.5,
        auprc=auprc if auprc is not None else pos_rate,
        pos_rate=pos_rate,
        status='TRAINED'
    )


# ------------------------- Main pipeline -------------------------

def main(csv_path="02_Processed_Data/FINAL_ENGINEERED_DATASET.csv", 
         exclude_leaking_cols=True,
         test_size=0.2,
         random_state=42,
         models_to_train=['xgboost', 'random_forest', 'lightgbm']):  # Train only best 3 for speed
    
    print("="*80)
    print("MULTIMORBIDITY MULTI-LABEL CLASSIFICATION")
    print("Training models for ALL 12 diseases")
    print("="*80)
    
    # Load dataset
    print("\n[1/7] Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"    ✓ Loaded {len(df):,} subjects with {len(df.columns)} columns")

    # Detect ALL target columns (diseases)
    print("\n[2/7] Detecting target columns (diseases)...")
    target_cols = detect_target_columns(df)
    
    if len(target_cols) == 0:
        print("    ❌ ERROR: No target columns found!")
        print("    Looking for columns starting with 'TARGET_' or disease keywords")
        return
    
    print(f"    ✓ Found {len(target_cols)} target diseases:")
    for i, col in enumerate(target_cols, 1):
        pos_rate = df[col].mean() if col in df.columns else 0
        print(f"       {i}. {col} (prevalence: {pos_rate:.2%})")

    # Get leaking columns to exclude
    leaking_cols = get_leaking_columns() if exclude_leaking_cols else []
    
    # Convert boolean columns to int
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    if bool_cols:
        print(f"\n[3/7] Converting {len(bool_cols)} boolean columns to int...")
        df[bool_cols] = df[bool_cols].astype(int)

    # Separate features from all targets
    print(f"\n[4/7] Preparing features...")
    all_exclude = target_cols + leaking_cols
    X = df.drop(columns=[col for col in all_exclude if col in df.columns])
    
    if exclude_leaking_cols and leaking_cols:
        actually_excluded = [c for c in leaking_cols if c in df.columns]
        print(f"    ✓ Excluded {len(actually_excluded)} potentially leaking columns")
    
    print(f"    ✓ Final feature count: {len(X.columns)}")

    # Train/test split (use first target for stratification)
    print(f"\n[5/7] Creating train-test split...")
    y_first = df[target_cols[0]]
    X_train, X_test, _, _ = train_test_split(
        X, y_first, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_first if len(np.unique(y_first))>1 else None
    )
    print(f"    ✓ Train set: {len(X_train):,} samples")
    print(f"    ✓ Test set: {len(X_test):,} samples")

    # Identify feature types
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"    ✓ Numeric features: {len(numeric_cols)}")
    print(f"    ✓ Categorical features: {len(categorical_cols)}")

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # ---------------- Train Models for ALL Targets ----------------
    print(f"\n[6/7] Training models for each disease...")
    print("="*80)
    
    all_results = []
    all_trained_models = {}
    
    # For each disease/target
    for target_idx, target_col in enumerate(target_cols, 1):
        print(f"\n{'='*80}")
        print(f"TARGET {target_idx}/{len(target_cols)}: {target_col}")
        print(f"{'='*80}")
        
        # Get target labels
        y_train = df.loc[X_train.index, target_col]
        y_test = df.loc[X_test.index, target_col]
        
        pos_rate = y_train.mean()
        print(f"  Prevalence in training: {pos_rate:.2%}")
        
        # Skip if no variance
        if pos_rate == 0 or pos_rate == 1:
            print(f"  ⚠ SKIPPED: Only one class present")
            continue
        
        # Calculate class weight
        scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1
        
        target_models = {}
        
        # Train each model type
        model_configs = []
        
        if 'xgboost' in models_to_train and USE_XGBOOST and xgb is not None:
            model_configs.append(('xgboost', Pipeline(steps=[
                ('pre', preprocessor), 
                ('clf', xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    objective='binary:logistic', eval_metric='auc',
                    random_state=random_state, n_jobs=-1
                ))
            ])))
        
        if 'random_forest' in models_to_train:
            model_configs.append(('random_forest', Pipeline(steps=[
                ('pre', preprocessor), 
                ('clf', RandomForestClassifier(
                    n_estimators=200, max_depth=10, class_weight='balanced',
                    random_state=random_state, n_jobs=-1
                ))
            ])))
        
        if 'logistic_regression' in models_to_train:
            model_configs.append(('logistic_regression', Pipeline(steps=[
                ('pre', preprocessor), 
                ('clf', LogisticRegression(max_iter=1000, random_state=random_state))
            ])))
        
        if 'neural_network' in models_to_train:
            model_configs.append(('neural_network_mlp', Pipeline(steps=[
                ('pre', preprocessor), 
                ('clf', MLPClassifier(
                    hidden_layer_sizes=(128,64), max_iter=500, 
                    random_state=random_state
                ))
            ])))
        
        if 'lightgbm' in models_to_train and lgb is not None:
            model_configs.append(('lightgbm', Pipeline(steps=[
                ('pre', preprocessor), 
                ('clf', lgb.LGBMClassifier(
                    n_estimators=500, random_state=random_state, 
                    n_jobs=-1, verbose=-1
                ))
            ])))
        
        if 'catboost' in models_to_train and CatBoostClassifier is not None:
            model_configs.append(('catboost', Pipeline(steps=[
                ('pre', preprocessor), 
                ('clf', CatBoostClassifier(
                    verbose=0, iterations=500, random_state=random_state
                ))
            ])))
        
        # Train all models for this target
        for model_name, model_pipeline in model_configs:
            print(f"\n  Training {model_name}...")
            result = train_and_eval_single_target(
                model_pipeline, X_train, X_test, y_train, y_test,
                model_name, target_col
            )
            all_results.append(result)
            
            # Store trained model
            if result['status'] == 'TRAINED':
                if target_col not in all_trained_models:
                    all_trained_models[target_col] = {}
                all_trained_models[target_col][model_name] = model_pipeline

    # ---------------- Save Results ----------------
    print(f"\n[7/7] Saving results...")
    print("="*80)
    
    results_df = pd.DataFrame(all_results)
    
    # Create output directory
    os.makedirs('models/multi_label', exist_ok=True)
    
    # Save comprehensive results
    results_df.to_csv('models/multi_label/all_results.csv', index=False)
    print(f"✓ Saved all results: models/multi_label/all_results.csv")
    
    # Save all trained models
    joblib.dump(all_trained_models, 'models/multi_label/all_models.joblib')
    print(f"✓ Saved all models: models/multi_label/all_models.joblib")
    
    # Save target list
    with open('models/multi_label/target_diseases.json', 'w') as f:
        json.dump(target_cols, f, indent=2)
    print(f"✓ Saved target list: models/multi_label/target_diseases.json")
    
    # ---------------- Generate Summary Report ----------------
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    # Summary by target
    print("PERFORMANCE BY DISEASE:")
    print("-" * 80)
    
    for target in target_cols:
        target_results = results_df[results_df['target'] == target]
        if len(target_results) > 0:
            best = target_results.nlargest(1, 'f1').iloc[0]
            print(f"\n{target}:")
            print(f"  Best model: {best['model_name']}")
            print(f"  F1: {best['f1']:.3f} | AUC: {best['roc_auc']:.3f} | " +
                  f"Precision: {best['precision']:.3f} | Recall: {best['recall']:.3f}")
    
    # Summary by model
    print(f"\n\n{'='*80}")
    print("AVERAGE PERFORMANCE BY MODEL:")
    print("-" * 80)
    
    trained_only = results_df[results_df['status'] == 'TRAINED']
    if len(trained_only) > 0:
        model_summary = trained_only.groupby('model_name').agg({
            'f1': 'mean',
            'roc_auc': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'accuracy': 'mean'
        }).round(3).sort_values('f1', ascending=False)
        
        print(model_summary)
        
        # Save model summary
        model_summary.to_csv('models/multi_label/model_summary.csv')
        print(f"\n✓ Saved model summary: models/multi_label/model_summary.csv")
    
    print(f"\n{'='*80}")
    print(f"MULTI-LABEL TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Trained {len(trained_only)} models across {len(target_cols)} diseases")
    print(f"\nFiles saved in: models/multi_label/")


if __name__ == '__main__':
    main(
        csv_path="D:/3rd Year Shits/SL ML/02_Processed_Data/FINAL_ENGINEERED_DATASET.csv",
        exclude_leaking_cols=True,
        test_size=0.2,
        random_state=42,
        models_to_train=['xgboost', 'random_forest', 'lightgbm']  # Train best 3 for speed
    )