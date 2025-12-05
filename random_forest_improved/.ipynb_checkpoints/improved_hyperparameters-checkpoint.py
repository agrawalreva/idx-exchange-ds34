"""
Improved Random Forest hyperparameters to reduce overfitting.

The model was overfitting because:
1. Target-encoded features (ZipMeanPrice, etc.) were dominating
2. Model was too complex (deep trees, many estimators)
3. Regularization was too weak

Solution: Increase regularization parameters
"""

# OLD hyperparameters (causing overfitting):
# n_estimators=300
# max_depth=20
# min_samples_split=5
# min_samples_leaf=2

# NEW hyperparameters (reduced overfitting):
IMPROVED_RF_PARAMS = {
    'n_estimators': 200,          # Reduced from 300
    'max_depth': 15,              # Reduced from 20
    'min_samples_split': 20,      # Increased from 5
    'min_samples_leaf': 10,       # Increased from 2
    'max_features': 'sqrt',
    'bootstrap': True,
    'oob_score': True,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': 0
}

# Use this in your notebook:
# rf_model = RandomForestRegressor(**IMPROVED_RF_PARAMS)

