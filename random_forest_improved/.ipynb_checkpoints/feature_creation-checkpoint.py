import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')


def calculate_euclidean_distance(lat1, lon1, lat2, lon2):
    """
    Calculate Euclidean distance between two points (simpler than Haversine).
    Good approximation for small distances in California.
    """
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)


def frequency_encode(series, min_freq=10):
    """
    Frequency encoding - different from target encoding.
    Maps values to their frequency in the dataset.
    """
    freq_map = series.value_counts().to_dict()
    # For low frequency values, use a default
    default_freq = min_freq
    return series.map(lambda x: freq_map.get(x, default_freq))


def mean_encode(train_series, train_target, test_series, smoothing=10):
    """
    Mean encoding with smoothing (different from KFold target encoding).
    """
    # Calculate mean for each category
    mean_map = train_target.groupby(train_series).mean()
    global_mean = train_target.mean()
    
    # Count frequency
    count_map = train_series.value_counts()
    
    # Align indices for proper calculation
    aligned_counts = count_map.reindex(mean_map.index, fill_value=0)
    
    # Smoothing: weighted average between category mean and global mean
    smoothed = (aligned_counts * mean_map + smoothing * global_mean) / (aligned_counts + smoothing)
    
    # Map to train and test
    train_encoded = train_series.map(smoothed).fillna(global_mean)
    test_encoded = test_series.map(smoothed).fillna(global_mean)
    
    return train_encoded, test_encoded


# California regional centers (different approach from city distances)
CA_REGIONS = {
    'Bay_Area_Center': (37.7749, -122.4194),  # San Francisco
    'LA_Metro_Center': (34.0522, -118.2437),  # Los Angeles
    'San_Diego_Region': (32.7157, -117.1611),  # San Diego
    'Central_Valley': (36.7378, -119.7871),   # Fresno
    'Sacramento_Area': (38.5816, -121.4944), # Sacramento
    'Orange_County': (33.7879, -117.8531),    # Anaheim
    'Inland_Empire': (34.0522, -117.3235),    # Riverside area
    'Wine_Country': (38.2975, -122.2869),    # Napa
    'Central_Coast': (36.6002, -121.8947),     # Monterey
    'Desert_Region': (33.8303, -116.5453)     # Palm Springs
}


def create_advanced_features(train_df, test_df):
    """
    Create advanced features using a different approach from juee's method.
    Focuses on regional patterns, frequency encoding, and different interactions.
    """
    print("Creating advanced features...")
    
    # Combine for some calculations
    train_y = train_df['ClosePrice'].copy()
    full_df = pd.concat([train_df.drop(columns=['ClosePrice']), test_df], 
                       axis=0, ignore_index=True)
    
    # ============================================================
    # 1. TEMPORAL FEATURES (Different approach)
    # ============================================================
    print("  - Temporal features")
    for df in [train_df, test_df]:
        if 'CloseDate' in df.columns:
            df['CloseYear'] = df['CloseDate'].dt.year
            df['CloseMonth'] = df['CloseDate'].dt.month
            df['CloseDayOfMonth'] = df['CloseDate'].dt.day
            df['CloseDayOfWeek'] = df['CloseDate'].dt.dayofweek
            df['IsWeekend'] = (df['CloseDayOfWeek'] >= 5).astype(int)
            
            # Quarter and fiscal quarter
            df['CloseQuarter'] = df['CloseDate'].dt.quarter
            df['FiscalQuarter'] = ((df['CloseMonth'] - 1) // 3 + 1)
            
            # Market seasonality (different from juee's season mapping)
            df['IsSpringMarket'] = df['CloseMonth'].isin([3, 4, 5]).astype(int)
            df['IsSummerMarket'] = df['CloseMonth'].isin([6, 7, 8]).astype(int)
            df['IsFallMarket'] = df['CloseMonth'].isin([9, 10, 11]).astype(int)
            df['IsWinterMarket'] = df['CloseMonth'].isin([12, 1, 2]).astype(int)
        
        # Time-based features
        if 'ListingContractDate' in df.columns and 'CloseDate' in df.columns:
            df['MarketDays'] = (df['CloseDate'] - df['ListingContractDate']).dt.days
            df['MarketDays'] = df['MarketDays'].clip(lower=0, upper=365)
            df['MarketWeeks'] = df['MarketDays'] / 7
            df['QuickSale_30days'] = (df['MarketDays'] <= 30).astype(int)
            df['LongSale_90days'] = (df['MarketDays'] >= 90).astype(int)
        
        if 'PurchaseContractDate' in df.columns and 'CloseDate' in df.columns:
            df['EscrowDays'] = (df['CloseDate'] - df['PurchaseContractDate']).dt.days
            df['EscrowDays'] = df['EscrowDays'].clip(lower=0, upper=120)
    
    # ============================================================
    # 2. TRANSFORMATIONS (Different from juee's log transforms)
    # ============================================================
    print("  - Transformations")
    for df in [train_df, test_df]:
        if 'LivingArea' in df.columns:
            df['SqrtLivingArea'] = np.sqrt(df['LivingArea'])
            df['LivingAreaSquared'] = df['LivingArea'] ** 2
        
        if 'BedroomsTotal' in df.columns:
            df['SqrtBedrooms'] = np.sqrt(df['BedroomsTotal'])
        
        if 'BathroomsTotalInteger' in df.columns:
            df['SqrtBathrooms'] = np.sqrt(df['BathroomsTotalInteger'])
        
        if 'PropertyAgeAtClose' in df.columns:
            df['SqrtAge'] = np.sqrt(df['PropertyAgeAtClose'].clip(lower=0))
            df['AgeSquared'] = df['PropertyAgeAtClose'] ** 2
    
    # ============================================================
    # 3. RATIO & DENSITY FEATURES (Different ratios)
    # ============================================================
    print("  - Ratio features")
    for df in [train_df, test_df]:
        if 'LivingArea' in df.columns:
            # Space efficiency features
            if 'BedroomsTotal' in df.columns:
                df['SqFtPerBedroom'] = df['LivingArea'] / (df['BedroomsTotal'] + 0.1)
                df['BedroomDensity'] = df['BedroomsTotal'] / (df['LivingArea'] + 1) * 1000
            
            if 'BathroomsTotalInteger' in df.columns:
                df['SqFtPerBathroom'] = df['LivingArea'] / (df['BathroomsTotalInteger'] + 0.1)
                df['BathroomDensity'] = df['BathroomsTotalInteger'] / (df['LivingArea'] + 1) * 1000
            
            if 'ParkingTotal' in df.columns:
                df['ParkingRatio'] = df['ParkingTotal'] / (df['LivingArea'] + 1) * 1000
            
            if 'GarageSpaces' in df.columns:
                df['GarageRatio'] = df['GarageSpaces'] / (df['LivingArea'] + 1) * 1000
        
        # Room composition
        if 'BedroomsTotal' in df.columns and 'BathroomsTotalInteger' in df.columns:
            df['TotalRooms'] = df['BedroomsTotal'] + df['BathroomsTotalInteger']
            df['BathToBedRatio'] = df['BathroomsTotalInteger'] / (df['BedroomsTotal'] + 0.1)
        
        # Story features
        if 'Stories' in df.columns and 'LivingArea' in df.columns:
            df['AreaPerStory'] = df['LivingArea'] / (df['Stories'] + 0.1)
            df['IsMultiStory'] = (df['Stories'] > 1).astype(int)
    
    # ============================================================
    # 4. GEOGRAPHICAL FEATURES (Different approach)
    # ============================================================
    print("  - Geographical features")
    
    # Distance to regional centers (different from city distances)
    for region_name, (reg_lat, reg_lon) in CA_REGIONS.items():
        train_df[f'DistTo_{region_name}'] = calculate_euclidean_distance(
            train_df['Latitude'], train_df['Longitude'], reg_lat, reg_lon
        )
        test_df[f'DistTo_{region_name}'] = calculate_euclidean_distance(
            test_df['Latitude'], test_df['Longitude'], reg_lat, reg_lon
        )
    
    # Find closest region
    dist_cols = [f'DistTo_{region}' for region in CA_REGIONS.keys()]
    dist_cols = [c for c in dist_cols if c in train_df.columns]
    if dist_cols:
        train_df['DistToClosestRegion'] = train_df[dist_cols].min(axis=1)
        test_df['DistToClosestRegion'] = test_df[dist_cols].min(axis=1)
    
    # Coordinate-based features (different from juee's approach)
    for df in [train_df, test_df]:
        df['LatitudeRounded'] = df['Latitude'].round(2)
        df['LongitudeRounded'] = df['Longitude'].round(2)
        df['CoordProduct'] = df['Latitude'] * df['Longitude']
        df['CoordSum'] = df['Latitude'] + df['Longitude']
        df['CoordDistance'] = np.sqrt(df['Latitude']**2 + df['Longitude']**2)
    
    # ============================================================
    # 5. CLUSTERING (Different method - Agglomerative)
    # ============================================================
    print("  - Clustering")
    coords = full_df[['Latitude', 'Longitude']].values
    
    # Agglomerative clustering (different from KMeans/DBSCAN)
    for n_clusters in [10, 20, 30]:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        full_df[f'RegionCluster_{n_clusters}'] = clusterer.fit_predict(coords)
        
        train_df[f'RegionCluster_{n_clusters}'] = full_df.iloc[:len(train_df)][f'RegionCluster_{n_clusters}'].values
        test_df[f'RegionCluster_{n_clusters}'] = full_df.iloc[len(train_df):][f'RegionCluster_{n_clusters}'].values
    
    # ============================================================
    # 6. ENCODING (Frequency and Mean encoding - different from KFold)
    # ============================================================
    print("  - Encoding")
    
    # Frequency encoding for categoricals
    for col in ['City', 'CountyOrParish', 'PostalCode']:
        if col in train_df.columns:
            train_freq = frequency_encode(train_df[col])
            test_freq = test_df[col].map(train_df[col].value_counts().to_dict()).fillna(10)
            train_df[f'{col}_Freq'] = train_freq
            test_df[f'{col}_Freq'] = test_freq
    
    # Mean encoding with smoothing
    for col in ['City', 'PostalCode', 'CountyOrParish']:
        if col in train_df.columns:
            train_encoded, test_encoded = mean_encode(
                train_df[col], train_y, test_df[col], smoothing=15
            )
            train_df[f'{col}_MeanPrice'] = train_encoded
            test_df[f'{col}_MeanPrice'] = test_encoded
    
    # Cluster mean encoding
    for n_clusters in [10, 20, 30]:
        col = f'RegionCluster_{n_clusters}'
        train_encoded, test_encoded = mean_encode(
            train_df[col], train_y, test_df[col], smoothing=10
        )
        train_df[f'{col}_MeanPrice'] = train_encoded
        test_df[f'{col}_MeanPrice'] = test_encoded
    
    # ============================================================
    # 7. INTERACTION FEATURES (Different combinations)
    # ============================================================
    print("  - Interaction features")
    for df in [train_df, test_df]:
        # Amenity combinations
        if 'PoolPrivateYN' in df.columns and 'ViewYN' in df.columns:
            df['PoolAndView'] = (df['PoolPrivateYN'] & df['ViewYN']).astype(int)
        
        if 'FireplaceYN' in df.columns and 'AttachedGarageYN' in df.columns:
            df['FireplaceAndGarage'] = (df['FireplaceYN'] & df['AttachedGarageYN']).astype(int)
        
        # Amenity count
        amenity_cols = ['PoolPrivateYN', 'ViewYN', 'FireplaceYN', 'AttachedGarageYN', 'NewConstructionYN']
        amenity_cols = [c for c in amenity_cols if c in df.columns]
        if amenity_cols:
            df['AmenityCount'] = df[amenity_cols].sum(axis=1)
        
        # Property quality score
        quality_score = 0
        if 'NewConstructionYN' in df.columns:
            quality_score += df['NewConstructionYN'].astype(int) * 3
        if 'ViewYN' in df.columns:
            quality_score += df['ViewYN'].astype(int) * 2
        if 'PoolPrivateYN' in df.columns:
            quality_score += df['PoolPrivateYN'].astype(int) * 2
        if 'FireplaceYN' in df.columns:
            quality_score += df['FireplaceYN'].astype(int)
        df['QualityScore'] = quality_score
        
        # Size-quality interactions
        if 'LivingArea' in df.columns and 'NewConstructionYN' in df.columns:
            df['NewConstructionSize'] = df['LivingArea'] * df['NewConstructionYN'].astype(int)
        
        if 'LivingArea' in df.columns and 'ViewYN' in df.columns:
            df['ViewSize'] = df['LivingArea'] * df['ViewYN'].astype(int)
    
    # ============================================================
    # 8. LOCAL MARKET FEATURES
    # ============================================================
    print("  - Local market features")
    # Postal code statistics (different approach)
    # Ensure PostalCode is string for consistent merging
    if 'PostalCode' in train_df.columns:
        train_df['PostalCode'] = train_df['PostalCode'].astype(str)
    if 'PostalCode' in test_df.columns:
        test_df['PostalCode'] = test_df['PostalCode'].astype(str)
    
    postal_stats = train_df.groupby('PostalCode')['ClosePrice'].agg([
        'mean', 'median', 'std', 'count', 'min', 'max'
    ]).reset_index()
    postal_stats.columns = ['PostalCode', 'ZipMeanPrice', 'ZipMedianPrice', 
                           'ZipStdPrice', 'ZipCount', 'ZipMinPrice', 'ZipMaxPrice']
    postal_stats['PostalCode'] = postal_stats['PostalCode'].astype(str)
    
    train_df = train_df.merge(postal_stats, on='PostalCode', how='left')
    test_df = test_df.merge(postal_stats, on='PostalCode', how='left')
    
    # Fill missing with global stats
    global_mean = train_y.mean()
    global_median = train_y.median()
    for col in ['ZipMeanPrice', 'ZipMedianPrice']:
        train_df[col] = train_df[col].fillna(global_mean)
        test_df[col] = test_df[col].fillna(global_mean)
    
    # Price relative to local market
    train_df['PriceToZipMean'] = train_y / (train_df['ZipMeanPrice'] + 1)
    test_df['PriceToZipMean'] = 1.0  # Placeholder
    
    # ============================================================
    # 9. CLEANUP
    # ============================================================
    print("  - Cleanup")
    cols_to_drop = [
        'PurchaseContractDate', 'ListingContractDate', 'CloseDate',
        'ContractStatusChangeDate', 'PostalCode', 'CountyOrParish', 'City'
    ]
    cols_to_drop = [c for c in cols_to_drop if c in train_df.columns]
    
    train_df = train_df.drop(columns=cols_to_drop)
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
    
    print(f"✓ Feature creation complete")
    print(f"  Train: {train_df.shape}")
    print(f"  Test:  {test_df.shape}")
    
    return train_df, test_df

