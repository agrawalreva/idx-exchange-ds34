"""
Solution to fix overfitting caused by target-encoded features.

Add this code to your notebook AFTER feature creation and BEFORE model training.
This will reduce the impact of target-encoded features that are causing overfitting.
"""

import numpy as np
import pandas as pd

def reduce_target_encoded_features(X_train, X_test, threshold=0.3):
    """
    Reduce the impact of target-encoded features that cause overfitting.
    
    Strategy: Remove or downweight features that are too highly correlated
    with target-encoded features, as they leak information.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    threshold : float
        Maximum correlation threshold for target-encoded features
        
    Returns:
    --------
    X_train_reduced, X_test_reduced : pd.DataFrame
        Features with reduced target-encoded feature impact
    removed_features : list
        List of features that were removed
    """
    
    # Identify target-encoded features
    target_encoded_cols = [col for col in X_train.columns if any(
        x in col for x in ['MeanPrice', 'ZipMean', 'ZipMedian', 'ZipStd', 
                          'ZipMin', 'ZipMax', 'PriceToZip']
    )]
    
    print(f"Found {len(target_encoded_cols)} target-encoded features")
    print(f"  Examples: {target_encoded_cols[:5]}")
    
    # Strategy 1: Keep only the most important target-encoded features
    # Remove redundant ones (keep only ZipMeanPrice and one MeanPrice per location type)
    features_to_remove = []
    
    # Remove redundant zip code statistics (keep only mean and median)
    zip_stats = [col for col in target_encoded_cols if 'Zip' in col]
    zip_stats_to_remove = [col for col in zip_stats if col not in ['ZipMeanPrice', 'ZipMedianPrice']]
    features_to_remove.extend(zip_stats_to_remove)
    
    # Remove PriceToZipMean (too direct a relationship)
    if 'PriceToZipMean' in X_train.columns:
        features_to_remove.append('PriceToZipMean')
    
    # Keep only one MeanPrice per location type (keep the most general one)
    mean_price_cols = [col for col in target_encoded_cols if 'MeanPrice' in col and 'Zip' not in col]
    if len(mean_price_cols) > 1:
        # Keep City_MeanPrice, remove others
        to_keep = 'City_MeanPrice' if 'City_MeanPrice' in mean_price_cols else mean_price_cols[0]
        features_to_remove.extend([col for col in mean_price_cols if col != to_keep])
    
    # Remove cluster mean prices (keep only one)
    cluster_mean_cols = [col for col in target_encoded_cols if 'RegionCluster' in col and 'MeanPrice' in col]
    if len(cluster_mean_cols) > 1:
        # Keep the one with most clusters (usually more general)
        to_keep = sorted(cluster_mean_cols, key=lambda x: int(x.split('_')[1]))[-1] if cluster_mean_cols else None
        if to_keep:
            features_to_remove.extend([col for col in cluster_mean_cols if col != to_keep])
    
    # Remove the identified features
    features_to_remove = [f for f in features_to_remove if f in X_train.columns]
    
    if features_to_remove:
        print(f"\nRemoving {len(features_to_remove)} redundant target-encoded features:")
        print(f"  {features_to_remove[:10]}")
        
        X_train_reduced = X_train.drop(columns=features_to_remove)
        X_test_reduced = X_test.drop(columns=[f for f in features_to_remove if f in X_test.columns])
        
        print(f"\nReduced features from {len(X_train.columns)} to {len(X_train_reduced.columns)}")
        return X_train_reduced, X_test_reduced, features_to_remove
    else:
        print("No redundant features to remove")
        return X_train, X_test, []


# Usage in notebook:
# X_train_reduced, X_test_reduced, removed = reduce_target_encoded_features(X_train, X_test)
# X_train = X_train_reduced
# X_test = X_test_reduced

