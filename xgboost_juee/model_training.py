import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import optuna
from xgboost import XGBRegressor
import joblib
import warnings
import sys
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
# Import your feature engineering module
try:
    from feature_engineering import build_features
except ImportError:
    print("ERROR: feature_engineering.py not found!")
    print("Make sure build_features() is defined in feature_engineering.py")
    sys.exit(1)

import cleaning_script as cs


# ===============================================
# Custom Metrics
# ===============================================

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (as percentage)"""
    return mean_absolute_percentage_error(y_true, y_pred) * 100

def mdape(y_true, y_pred):
    """Median Absolute Percentage Error (as percentage)"""
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100


# ===============================================
# Load Data
# ===============================================
def load_data():
    """Load pre-cleaned train and test datasets"""
    print("Loading data...")
    
    try:
        # Identify date columns that exist
        train_sample = pd.read_csv("train_cleaned.csv", nrows=1)
        date_cols = [col for col in ["CloseDate", "ListingContractDate", "PurchaseContractDate", "ContractStatusChangeDate"] 
                     if col in train_sample.columns]
        
        train_cleaned, stats, mechs = cs.adaptive_clean_real_estate_data(pd.read_csv("combined_raw_data.csv", parse_dates=date_cols if date_cols else False), fit = True)
        test_cleaned, _, _= cs.adaptive_clean_real_estate_data(pd.read_csv("testing_raw_data.csv", parse_dates=date_cols if date_cols else False), fit = False, stats = stats, mechs = mechs)
        
        print(f"  Train: {train_cleaned.shape}")
        print(f"  Test:  {test_cleaned.shape}")
        
        return train_cleaned, test_cleaned
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Make sure train_cleaned.csv and test_cleaned.csv exist in the current directory")
        sys.exit(1)


# ===============================================
# Encode Categorical Features
# ===============================================
def encode_features(train_df, test_df, categorical_cols=None):
    """
    Label encode categorical features consistently between train and test.
    
    Returns:
        train_encoded, test_encoded, encoders
    """
    if categorical_cols is None:
        # Auto-detect categorical columns
        categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        # Filter out date-like columns
        categorical_cols = [col for col in categorical_cols 
                           if 'date' not in col.lower() and 'Date' not in col]
    
    # Filter to only columns that exist in both datasets
    categorical_cols = [col for col in categorical_cols 
                       if col in train_df.columns and col in test_df.columns]
    
    if len(categorical_cols) == 0:
        print("  No categorical features to encode")
        return train_df, test_df, {}
    
    print(f"\nEncoding {len(categorical_cols)} categorical features...")
    
    encoders = {}
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    for col in categorical_cols:
        # Fit encoder on training data
        le = LabelEncoder()
        train_encoded[f'{col}_encoded'] = le.fit_transform(train_df[col].astype(str))
        encoders[col] = le
        
        # Transform test data, handling unseen categories
        def safe_transform(value):
            value_str = str(value)
            if value_str in le.classes_:
                return le.transform([value_str])[0]
            else:
                return -1  # Unseen category marker
        
        test_encoded[f'{col}_encoded'] = test_df[col].apply(safe_transform)
        
        unseen_count = (test_encoded[f'{col}_encoded'] == -1).sum()
        if unseen_count > 0:
            print(f"  ⚠️  {col}: {unseen_count} unseen categories in test (marked as -1)")
        else:
            print(f"  ✓ {col}: {len(le.classes_)} unique values")
        
        # Drop original categorical column
        train_encoded = train_encoded.drop(columns=[col])
        test_encoded = test_encoded.drop(columns=[col])
    
    return train_encoded, test_encoded, encoders


# ===============================================
# Prepare Data
# ===============================================
def prepare_data():
    """
    Load data, build features, split into X/y.
    
    Returns:
        X_train, y_train, X_test, y_test (or None)
    """
    train_raw, test_raw = load_data()
    
    print("\nBuilding features...")
    try:
        train_fe, test_fe = build_features(train_raw, test_raw)
    except Exception as e:
        print(f"ERROR in build_features(): {e}")
        raise
    
    print(f"  Train shape after feature engineering: {train_fe.shape}")
    print(f"  Test shape after feature engineering:  {test_fe.shape}")
    
    # Verify ClosePrice exists
    if "ClosePrice" not in train_fe.columns:
        raise ValueError("ClosePrice column not found in training data!")
    
    # Check for any remaining categorical columns (should be none)
    train_cats = train_fe.select_dtypes(include=['object', 'category']).columns.tolist()
    test_cats = test_fe.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if train_cats or test_cats:
        print(f"\n⚠️  Unexpected categorical columns found:")
        if train_cats:
            print(f"  Train: {train_cats}")
        if test_cats:
            print(f"  Test: {test_cats}")
        raise ValueError("Feature engineering should return only numeric features!")
    
    print("  ✓ All features are numeric")
    
    train_encoded, test_encoded = train_fe, test_fe
    
    # Drop columns that shouldn't be predictors (should already be dropped by feature engineering)
    drop_cols = []
    for col in ["CloseDate", "ListingContractDate", "PurchaseContractDate", "ContractStatusChangeDate"]:
        if col in train_encoded.columns:
            drop_cols.append(col)
            print(f"  ⚠️  Found unexpected date column: {col}")
    
    if drop_cols:
        print(f"  Dropping {len(drop_cols)} date columns (should be handled in feature engineering)")
    
    # Prepare training data
    X_train = train_encoded.drop(columns=drop_cols + ["ClosePrice"], errors="ignore")
    y_train = train_encoded["ClosePrice"]
    
    # Prepare test data
    X_test = test_encoded.drop(columns=drop_cols, errors="ignore")
    
    # Check if test has ClosePrice (for evaluation)
    if "ClosePrice" in test_encoded.columns:
        y_test = test_encoded["ClosePrice"]
        X_test = X_test.drop(columns=["ClosePrice"], errors="ignore")
    else:
        y_test = None
        print("  Note: Test set has no ClosePrice column (prediction only)")
    
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    # Ensure identical features
    if list(X_train.columns) != list(X_test.columns):
        print("\n⚠️  WARNING: Feature mismatch between train and test!")
        train_only = set(X_train.columns) - set(X_test.columns)
        test_only = set(X_test.columns) - set(X_train.columns)
        if train_only:
            print(f"  Features only in train: {train_only}")
        if test_only:
            print(f"  Features only in test: {test_only}")
        raise ValueError("Feature mismatch detected! Check build_features() output.")
    
    print(f"  ✓ Feature alignment verified: {len(X_train.columns)} features")
    
    return X_train, y_train, X_test, y_test


# ===============================================
# Optuna Objective
# ===============================================
def objective(trial, X, y, n_folds=3):
    """
    Optuna objective function with k-fold cross-validation.
    Optimizes for R² score (maximize).
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 3),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 3),
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1
    }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBRegressor(**params)
        
        model.fit(X_tr, y_tr, verbose=False)

        preds = model.predict(X_val)
        fold_r2 = r2_score(y_val, preds)
        fold_scores.append(fold_r2)
        
        # Report intermediate score for pruning
        trial.report(np.mean(fold_scores), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(fold_scores)


# ===============================================
# Train Final Model
# ===============================================
def train_final_model(X, y, best_params, test_size=0.2, random_state=42):
    """
    Train final XGBoost model with validation split.
    
    Returns:
        model: Trained XGBoost model
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL")
    print("="*60)
    
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train size: {X_train.shape[0]}")
    print(f"Validation size: {X_val.shape[0]}")
    
    # Ensure n_estimators exists
    if "n_estimators" not in best_params:
        best_params["n_estimators"] = 1000
    
    # Add n_jobs for parallel processing
    best_params["n_jobs"] = -1
    
    # Initialize model
    model = XGBRegressor(**best_params)
    
    print("\nTraining model...")
    model.fit(X_train, y_train, verbose=False)
    
    # Predictions on both sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Compute metrics
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    train_mape = mape(y_train, y_train_pred)
    val_mape = mape(y_val, y_val_pred)
    
    train_mdape = mdape(y_train, y_train_pred)
    val_mdape = mdape(y_val, y_val_pred)
    
    # Print metrics
    print("FINAL MODEL METRICS")
    print(f"\nTraining Set:")
    print(f"  R²:     {train_r2:.4f}")
    print(f"  MAPE:   {train_mape:.2f}%")
    print(f"  MdAPE:  {train_mdape:.2f}%")
    
    print(f"\nValidation Set:")
    print(f"  R²:     {val_r2:.4f}")
    print(f"  MAPE:   {val_mape:.2f}%")
    print(f"  MdAPE:  {val_mdape:.2f}%")
    
    print(f"\nOverfitting Check:")
    print(f"  R² difference: {train_r2 - val_r2:.4f}")
    if train_r2 - val_r2 > 0.1:
        print("  ⚠️  Significant overfitting detected")
    elif train_r2 - val_r2 > 0.05:
        print("  ⚠️  Mild overfitting detected")
    else:
        print("  ✓  Good generalization")
    
    metrics = {
        "train_r2": train_r2,
        "val_r2": val_r2,
        "train_mape": train_mape,
        "val_mape": val_mape,
        "train_mdape": train_mdape,
        "val_mdape": val_mdape
    }
    
    return model, metrics


# ===============================================
# Evaluate Test Set
# ===============================================
def evaluate_test_set(model, X_test, y_test):
    """
    Generate predictions on test set and evaluate if ground truth available.
    """
    print("TEST SET EVALUATION")
    
    # Make predictions
    test_preds = model.predict(X_test)
    
    if y_test is not None:
        # Calculate metrics
        test_r2 = r2_score(y_test, test_preds)
        test_mape = mape(y_test, test_preds)
        test_mdape = mdape(y_test, test_preds)
        
        print(f"\nTest Set Metrics:")
        print(f"  R²:     {test_r2:.4f}")
        print(f"  MAPE:   {test_mape:.2f}%")
        print(f"  MdAPE:  {test_mdape:.2f}%")
        
        return test_preds, {
            "test_r2": test_r2,
            "test_mape": test_mape,
            "test_mdape": test_mdape
        }
    else:
        print("No ground truth available (prediction only)")
        return test_preds, {}


# ===============================================
# Main Pipeline
# ===============================================
def main(n_trials=40, n_cv_folds=3, skip_tuning=False):
    """
    Main training pipeline.
    
    Parameters:
    -----------
    n_trials : int
        Number of Optuna trials for hyperparameter tuning
    n_cv_folds : int
        Number of cross-validation folds
    skip_tuning : bool
        If True, use default parameters and skip Optuna tuning
        
    Returns:
    --------
    model : XGBRegressor
        Trained model
    metrics : dict
        All evaluation metrics
    """
    

    print("XGBOOST REAL ESTATE PRICE PREDICTION PIPELINE")
    
    
    # Step 1: Prepare data
    X_train, y_train, X_test, y_test = prepare_data()
    
    # Step 2: Hyperparameter tuning (optional)
    if not skip_tuning:
        print("\n" + "="*60)
        print(f"HYPERPARAMETER TUNING ({n_trials} trials, {n_cv_folds}-fold CV)")
        print("="*60)
        
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, n_folds=n_cv_folds), 
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        print(f"\n✓ Tuning complete!")
        print(f"Best R²: {study.best_value:.4f}")
        print(f"\nBest Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
    else:
        print("\n⚠️  Skipping hyperparameter tuning, using defaults")
        best_params = {
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "random_state": 42,
            "tree_method": "hist"
        }
    
    # Step 3: Train final model
    final_model, train_metrics = train_final_model(X_train, y_train, best_params)
    
    # Step 4: Evaluate on test set
    test_preds, test_metrics = evaluate_test_set(final_model, X_test, y_test)
    
    # Step 5: Save artifacts
    print("\n" + "="*60)
    print("SAVING ARTIFACTS")
    print("="*60)
    
    joblib.dump(final_model, "xgb_final_model.pkl")
    print("✓ Model saved: xgb_final_model.pkl")
    
    joblib.dump(best_params, "best_params.pkl")
    print("✓ Parameters saved: best_params.pkl")
    
    # Save predictions
    pred_df = pd.DataFrame({"predictions": test_preds})
    if y_test is not None:
        pred_df["actual"] = y_test.values
    pred_df.to_csv("test_predictions.csv", index=False)
    print("✓ Predictions saved: test_predictions.csv")
    
    # Combine all metrics
    all_metrics = {**train_metrics, **test_metrics}
    pd.DataFrame([all_metrics]).to_csv("model_metrics.csv", index=False)
    print("✓ Metrics saved: model_metrics.csv")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    
    return final_model, all_metrics


if __name__ == "__main__":
    # Run with default settings
    model, metrics = main(
        n_trials=40,
        n_cv_folds=3,
        skip_tuning=False
    )