import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans


# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two lat/lon points in kilometers"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def kfold_target_encode(train_series, target, test_series, n_splits=5):
    """
    K-fold target encoding to prevent leakage.
    Returns out-of-fold encoded training series and encoded test series.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = pd.Series(np.nan, index=train_series.index)
    test_encoded = pd.Series(np.zeros(len(test_series)), index=test_series.index)

    for train_idx, valid_idx in kf.split(train_series):
        tr, val = train_series.iloc[train_idx], train_series.iloc[valid_idx]
        target_tr = target.iloc[train_idx]

        means = target_tr.groupby(tr).mean()
        oof.iloc[valid_idx] = val.map(means)
        test_encoded += test_series.map(means).fillna(means.mean())

    test_encoded /= n_splits
    overall_mean = target.mean()
    return oof.fillna(overall_mean), test_encoded.fillna(overall_mean)


# --------------------------------------------------------
# Coastal Reference Points (Major California Coast Cities)
# --------------------------------------------------------

COAST_POINTS = [
    (32.543, -117.124),  # San Diego
    (33.600, -117.900),  # Laguna Beach
    (34.420, -119.698),  # Santa Barbara
    (34.281, -119.300),  # Ventura
    (34.014, -118.496),  # Santa Monica
    (36.600, -121.900),  # Monterey
    (36.778, -122.000),  # Santa Cruz
    (37.774, -122.510),  # San Francisco
    (38.300, -123.050),  # Sonoma Coast
    (40.440, -124.410),  # Humboldt
]

def min_distance_to_coast(df, coast_points):
    """Calculate minimum distance to any coastal point"""
    all_distances = []
    for lat_c, lon_c in coast_points:
        dist = haversine_distance(df["Latitude"], df["Longitude"], lat_c, lon_c)
        all_distances.append(dist)
    return np.column_stack(all_distances).min(axis=1)


# --------------------------------------------------------
# MAIN FEATURE ENGINEERING PIPELINE
# --------------------------------------------------------

def build_features(train_cleaned, test_cleaned):
    """
    Build features for real estate price prediction.
    
    Parameters:
    -----------
    train_cleaned : pd.DataFrame
        Cleaned training data with ClosePrice
    test_cleaned : pd.DataFrame
        Cleaned test data with or without ClosePrice
        
    Returns:
    --------
    train_fe : pd.DataFrame
        Training data with engineered features
    test_fe : pd.DataFrame
        Test data with engineered features
    """
    
    # Work on copies to avoid modifying originals
    train = train_cleaned.copy()
    test = test_cleaned.copy()
    
    print("Building features...")
    
    # --------------------------------------------------------
    # 1. DATE FEATURES
    # --------------------------------------------------------
    print("  - Temporal features")
    for df in [train, test]:
        df["CloseYear"] = df["CloseDate"].dt.year
        df["CloseMonth"] = df["CloseDate"].dt.month
        df["CloseQuarter"] = df["CloseDate"].dt.quarter
        
        # Encode season as numeric (0-3) instead of string
        season_map = {
            12: 0, 1: 0, 2: 0,      # Winter
            3: 1, 4: 1, 5: 1,        # Spring
            6: 2, 7: 2, 8: 2,        # Summer
            9: 3, 10: 3, 11: 3,      # Fall
        }
        df["CloseSeason"] = df["CloseMonth"].map(season_map)
        
        # Time on market features
        df["DaysOnMarket"] = (df["CloseDate"] - df["ListingContractDate"]).dt.days
        df["TimeFromOfferToClose"] = (df["CloseDate"] - df["PurchaseContractDate"]).dt.days
        df["TimeFromListingToOffer"] = (df["PurchaseContractDate"] - df["ListingContractDate"]).dt.days
        
        # Handle negative values (data quality issues)
        df["DaysOnMarket"] = df["DaysOnMarket"].clip(lower=0)
        df["TimeFromOfferToClose"] = df["TimeFromOfferToClose"].clip(lower=0)
        df["TimeFromListingToOffer"] = df["TimeFromListingToOffer"].clip(lower=0)

    # --------------------------------------------------------
    # 2. LOG TRANSFORMS & AGE BUCKETS
    # --------------------------------------------------------
    print("  - Log transforms and age buckets")
    for df in [train, test]:
        df["LogLivingArea"] = np.log1p(df["LivingArea"])
        df["LogLotSizeSF"] = np.log1p(df["LotSizeSquareFeet"])
        
        # Convert age bucket to numeric codes instead of categorical
        df["AgeBucket_0_5"] = (df["PropertyAgeAtClose"] <= 5).astype(int)
        df["AgeBucket_5_20"] = ((df["PropertyAgeAtClose"] > 5) & (df["PropertyAgeAtClose"] <= 20)).astype(int)
        df["AgeBucket_20_50"] = ((df["PropertyAgeAtClose"] > 20) & (df["PropertyAgeAtClose"] <= 50)).astype(int)
        df["AgeBucket_50plus"] = (df["PropertyAgeAtClose"] > 50).astype(int)

    # --------------------------------------------------------
    # 3. RATIO FEATURES
    # --------------------------------------------------------
    print("  - Ratio features")
    def add_ratios(df):
        # Avoid division by zero
        df["LotSizeToLivingArea"] = df["LotSizeSquareFeet"] / (df["LivingArea"] + 1)
        df["BathsPerSqFt"] = df["BathroomsTotalInteger"] / (df["LivingArea"] + 1)
        df["BedsPerSqFt"] = df["BedroomsTotal"] / (df["LivingArea"] + 1)
        df["RoomsPerSqFt"] = (df["BedroomsTotal"] + df["BathroomsTotalInteger"]) / (df["LivingArea"] + 1)
        df["ParkingPerSqFt"] = df["ParkingTotal"] / (df["LivingArea"] + 1)
        df["GarageSpacesPerBedroom"] = df["GarageSpaces"] / (df["BedroomsTotal"] + 1)
        df["Lat_Long_Interaction"] = df["Latitude"] * df["Longitude"]
        df["IsSingleStory"] = (df["Stories"] == 1).astype(int)
        df["IsTwoStory"] = (df["Stories"] == 2).astype(int)

    add_ratios(train)
    add_ratios(test)

    # --------------------------------------------------------
    # 4. BOOLEAN INTERACTION FEATURES
    # --------------------------------------------------------
    print("  - Boolean interactions")
    for df in [train, test]:
        df["HasPoolWithView"] = (df["PoolPrivateYN"] & df["ViewYN"]).astype(int)
        df["HasGarageAndFireplace"] = (df["FireplaceYN"] & df["AttachedGarageYN"]).astype(int)
        
        df["LuxuryAmenityScore"] = (
            df["PoolPrivateYN"].astype(int)
            + df["ViewYN"].astype(int)
            + df["FireplaceYN"].astype(int)
            + df["NewConstructionYN"].astype(int)
            + df["AttachedGarageYN"].astype(int)
        )

    # --------------------------------------------------------
    # 5. DISTANCE TO COAST
    # --------------------------------------------------------
    print("  - Coastal distance")
    train["DistToCoast_km"] = min_distance_to_coast(train, COAST_POINTS)
    test["DistToCoast_km"] = min_distance_to_coast(test, COAST_POINTS)
    
    # Create coastal proximity indicator
    for df in [train, test]:
        df["IsCoastal"] = (df["DistToCoast_km"] < 10).astype(int)  # Within 10km

    # --------------------------------------------------------
    # 6. KMEANS GEO CLUSTERS
    # --------------------------------------------------------
    print("  - Geographic clustering")
    # Combine for clustering (excluding target)
    train_coords = train[["Latitude", "Longitude"]]
    test_coords = test[["Latitude", "Longitude"]]
    all_coords = pd.concat([train_coords, test_coords], ignore_index=True)
    
    km = KMeans(n_clusters=20, random_state=42, n_init=10)
    all_clusters = km.fit_predict(all_coords)
    
    train["GeoCluster"] = all_clusters[:len(train)]
    test["GeoCluster"] = all_clusters[len(train):]

    # --------------------------------------------------------
    # 7. TARGET ENCODING (No Leakage via K-Fold)
    # --------------------------------------------------------
    print("  - Target encoding (k-fold)")
    y = train["ClosePrice"]
    
    for col in ["City", "PostalCode", "CountyOrParish"]:
        tr_enc, te_enc = kfold_target_encode(train[col], y, test[col])
        train[f"{col}_TE"] = tr_enc
        test[f"{col}_TE"] = te_enc

    # --------------------------------------------------------
    # 8. LOCAL PRICE PER SQFT BENCHMARKS (No Leakage)
    # --------------------------------------------------------
    print("  - Local pricing benchmarks")
    # Calculate PPSF only on training data
    train_ppsf = train["ClosePrice"] / train["LivingArea"]
    
    # Compute medians from training only
    ppsf_zip = train_ppsf.groupby(train["PostalCode"]).median()
    ppsf_city = train_ppsf.groupby(train["City"]).median()
    ppsf_county = train_ppsf.groupby(train["CountyOrParish"]).median()
    
    # Apply to both train and test
    for df in [train, test]:
        df["ZIP_MedianPPSF"] = df["PostalCode"].map(ppsf_zip).fillna(ppsf_city.median())
        df["City_MedianPPSF"] = df["City"].map(ppsf_city).fillna(ppsf_city.median())
        df["County_MedianPPSF"] = df["CountyOrParish"].map(ppsf_county).fillna(ppsf_county.median())

    # --------------------------------------------------------
    # 9. MARKET VELOCITY SIGNALS (No Leakage)
    # --------------------------------------------------------
    print("  - Market velocity indicators")
    # Create month string for aggregation
    train["MonthStr"] = train["CloseDate"].dt.to_period("M").astype(str)
    
    # Calculate from training data only
    sales_count = train.groupby(["PostalCode", "MonthStr"]).size()
    median_dom = train.groupby(["PostalCode", "MonthStr"])["DaysOnMarket"].median()
    
    # Apply to both datasets
    for df in [train, test]:
        df["MonthStr"] = df["CloseDate"].dt.to_period("M").astype(str)
        
        # Map sales count
        df["ZIP_SalesCount"] = df.apply(
            lambda row: sales_count.get((row["PostalCode"], row["MonthStr"]), 0), 
            axis=1
        )
        
        # Map median days on market
        df["ZIP_MedianDOM"] = df.apply(
            lambda row: median_dom.get((row["PostalCode"], row["MonthStr"]), median_dom.median()), 
            axis=1
        )

    # --------------------------------------------------------
    # 10. CLEANUP - Drop columns that shouldn't be features
    # --------------------------------------------------------
    print("  - Cleanup")
    
    # Drop raw categorical columns (we have target-encoded versions)
    columns_to_drop = [
        "City", "PostalCode", "CountyOrParish",  # Have _TE versions
        "MonthStr",  # Temporary helper column
        "CloseDate", "ListingContractDate", "PurchaseContractDate",  # Date columns
        "ContractStatusChangeDate"  # If it exists
    ]
    
    # Only drop columns that exist
    train = train.drop(columns=[col for col in columns_to_drop if col in train.columns])
    test = test.drop(columns=[col for col in columns_to_drop if col in test.columns])
    
    print(f"✓ Feature engineering complete")
    print(f"  Train: {train.shape}")
    print(f"  Test:  {test.shape}")
    
    # Verify no categorical columns remain (except ClosePrice)
    train_cats = train.select_dtypes(include=['object', 'category']).columns.tolist()
    test_cats = test.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if train_cats:
        print(f"  ⚠️  Warning: Categorical columns in train: {train_cats}")
    if test_cats:
        print(f"  ⚠️  Warning: Categorical columns in test: {test_cats}")
    
    return train, test


if __name__ == "__main__":
    # Test the function
    print("Feature engineering module loaded successfully!")
    print("\nUsage:")
    print("  from feature_engineering import build_features")
    print("  train_fe, test_fe = build_features(train_raw, test_raw)")