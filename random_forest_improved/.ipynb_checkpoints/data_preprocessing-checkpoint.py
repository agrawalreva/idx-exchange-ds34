import pandas as pd
import numpy as np

def preprocess_real_estate_data(df):
    """
    Preprocess real estate dataset - unique approach focusing on data quality.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw real estate dataset
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataset ready for feature engineering
    """
    df = df.copy()
    
    # Step 1: Filter for California, Residential, SingleFamilyResidence
    if 'StateOrProvince' in df.columns:
        df = df[df['StateOrProvince'] == 'CA']
    if 'PropertyType' in df.columns:
        df = df[df['PropertyType'] == 'Residential']
    if 'PropertySubType' in df.columns:
        df = df[df['PropertySubType'] == 'SingleFamilyResidence']
    
    # Step 2: Drop columns with excessive missing data (>80%)
    missing_threshold = 0.80
    high_missing_cols = df.columns[df.isna().mean() > missing_threshold].tolist()
    df = df.drop(columns=high_missing_cols)
    
    # Step 3: Drop identifier and agent-related columns
    drop_cols = [
        'ListingKey', 'ListingKeyNumeric', 'ListingId', 'UnparsedAddress',
        'ListPrice', 'OriginalListPrice', 'DaysOnMarket', 'AssociationFee',
        'AssociationFeeFrequency', 'HighSchoolDistrict', 'BuyerAgentAOR',
        'ListAgentAOR', 'BuyerOfficeAOR', 'ListAgentEmail', 'ListAgentFirstName',
        'ListAgentLastName', 'ListAgentFullName', 'BuyerAgentFirstName',
        'BuyerAgentLastName', 'BuyerAgentMlsId', 'ListOfficeName',
        'BuyerOfficeName', 'SubdivisionName', 'Levels', 'Flooring',
        'MLSAreaMajor', 'StreetNumberNumeric', 'MainLevelBedrooms', 'CoListOfficeName'
    ]
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Step 4: Handle ClosePrice outliers (winsorize at 1st and 99th percentile)
    if 'ClosePrice' in df.columns:
        lower_bound = df['ClosePrice'].quantile(0.01)
        upper_bound = df['ClosePrice'].quantile(0.99)
        df['ClosePrice'] = df['ClosePrice'].clip(lower=lower_bound, upper=upper_bound)
        df = df.dropna(subset=['ClosePrice'])
    
    # Step 5: Convert date columns
    date_cols = ['PurchaseContractDate', 'ListingContractDate', 'ContractStatusChangeDate', 'CloseDate']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Step 6: Fill missing values with domain knowledge
    if 'Stories' in df.columns:
        df['Stories'] = df['Stories'].fillna(1.0)
    
    if 'GarageSpaces' in df.columns and 'ParkingTotal' in df.columns:
        df['GarageSpaces'] = df['GarageSpaces'].fillna(df['ParkingTotal'])
    
    if 'LivingArea' in df.columns:
        if df['LivingArea'].isna().mean() < 0.02:
            df = df.dropna(subset=['LivingArea'])
        else:
            df['LivingArea'] = df['LivingArea'].fillna(df['LivingArea'].median())
    
    if 'YearBuilt' in df.columns and 'CloseDate' in df.columns:
        df['YearBuilt'] = df['YearBuilt'].fillna(df['YearBuilt'].median())
        if df['CloseDate'].dtype == 'datetime64[ns]':
            df['PropertyAgeAtClose'] = df['CloseDate'].dt.year - df['YearBuilt']
    
    if 'City' in df.columns and 'CountyOrParish' in df.columns:
        df['City'] = df['City'].fillna(df['CountyOrParish'])
    
    if 'ParkingTotal' in df.columns:
        df['ParkingTotal'] = df['ParkingTotal'].fillna(0)
    
    # Step 7: Handle coordinates - fill with postal code centroids
    if all(col in df.columns for col in ['Latitude', 'Longitude', 'PostalCode']):
        postal_centroids = (
            df[['PostalCode', 'Latitude', 'Longitude']]
            .dropna(subset=['PostalCode'])
            .groupby('PostalCode')
            .agg(centroid_lat=('Latitude', 'median'), centroid_lon=('Longitude', 'median'))
        )
        df['Latitude'] = df['Latitude'].fillna(df['PostalCode'].map(postal_centroids['centroid_lat']))
        df['Longitude'] = df['Longitude'].fillna(df['PostalCode'].map(postal_centroids['centroid_lon']))
    
    # Step 8: Convert boolean columns
    bool_cols = ['ViewYN', 'PoolPrivateYN', 'AttachedGarageYN', 'FireplaceYN', 'NewConstructionYN']
    for col in bool_cols:
        if col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].fillna(False)
            else:
                # Convert various formats to boolean
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = df[col].isin(['true', 't', 'yes', 'y', '1']).fillna(False)
    
    # Step 9: Fill remaining numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0 and df[col].isna().mean() < 0.10:
            df[col] = df[col].fillna(df[col].median())
    
    # Step 10: Final cleanup - drop rows with critical missing values
    critical_cols = ['LivingArea', 'Latitude', 'Longitude', 'CloseDate']
    critical_cols = [c for c in critical_cols if c in df.columns]
    df = df.dropna(subset=critical_cols)
    
    return df

