import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import optuna
from xgboost import XGBRegressor
import joblib

from feature_engineering import build_features


# ===============================================
# Custom Metrics
# ===============================================

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

def mdape(y_true, y_pred):
    return np.median(np.abs((y_true - y_pred) / y_true))


# ===============================================
# Load Raw Cleaned CSVs
# ===============================================
def load_data():
    train_raw = pd.read_csv("train_cleaned.csv", parse_dates=["CloseDate", "ListingContractDate", "PurchaseContractDate"])
    test_raw = pd.read_csv("test_cleaned.csv", parse_dates=["CloseDate", "ListingContractDate", "PurchaseContractDate"])
    return train_raw, test_raw


# ===============================================
# Build Features
# ===============================================
def prepare_data():
    train_raw, test_raw = load_data()

    train_fe, test_fe = build_features(train_raw, test_raw)

    # Drop leakage-prone columns that shouldn't be used as predictors
    drop_cols = ["ClosePrice", "CloseDate", "ListingContractDate", "PurchaseContractDate"]
    X = train_fe.drop(columns=drop_cols, errors="ignore")
    y = train_fe["ClosePrice"]

    return X, y, test_fe.drop(columns=["ClosePrice"], errors="ignore")


# ===============================================
# Optuna Objective (5-fold CV)
# ===============================================
def objective(trial, X, y):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 3),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 3),
        "random_state": 42,
        "tree_method": "hist"
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []

    for train_idx, valid_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBRegressor(**params)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=50
        )

        preds = model.predict(X_val)
        r2_scores.append(r2_score(y_val, preds))

        # Allow pruning
        trial.report(np.mean(r2_scores), len(r2_scores))
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(r2_scores)


# Train Final Model

def train_final_model(X, y, best_params):
    model = XGBRegressor(**best_params)

    # Final train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=50
    )

    preds = model.predict(X_val)

    # Compute metrics
    r2 = r2_score(y_val, preds)
    mape_val = mape(y_val, preds)
    mdape_val = mdape(y_val, preds)

    print("\n=============================")
    print(" FINAL MODEL METRICS")
    print("=============================")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape_val:.4f}")
    print(f"MdAPE: {mdape_val:.4f}")

    return model


# Main Pipeline

def main():

    print("Building features...")
    X, y, X_test = prepare_data()

    print(f"Train shape: {X.shape}")

    # ---- OPTUNA HYPERPARAM TUNING ----
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, X, y), n_trials=40)

    best_params = study.best_params
    print("Best Parameters:", best_params)

    # ---- TRAIN FINAL MODEL ----
    final_model = train_final_model(X, y, best_params)

    # Save model
    joblib.dump(final_model, "xgb_final_model.pkl")
    print("\nModel saved as xgb_final_model.pkl")


if __name__ == "__main__":
    main()
