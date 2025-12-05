import pandas as pd
import numpy as np

def clean_real_estate_data(df):
    """
    Clean real estate dataset to match the cleaned combined_dataset structure.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw real estate dataset
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataset with 23 columns
    """
    df = df.copy()
    
    # Step 1: Drop columns with >75% missing values
    threshold = 0.75
    cols_to_drop = df.columns[df.isna().mean() > threshold].tolist()
    df = df.drop(columns=cols_to_drop)
    
    # Step 2: Filter for CA, Residential, SingleFamilyResidence
    if 'StateOrProvince' in df.columns:
        df = df[df['StateOrProvince'] == 'CA']
    if 'PropertyType' in df.columns:
        df = df[df['PropertyType'] == 'Residential']
    if 'PropertySubType' in df.columns:
        df = df[df['PropertySubType'] == 'SingleFamilyResidence']
    
    # Step 3: Drop zero variance columns (after filtering)
    zero_variance_cols = df.columns[df.nunique() <= 1].tolist()
    df = df.drop(columns=zero_variance_cols)
    
    # Step 4: Drop specific columns (agent info, listing info, etc.)
    more_columns_to_drop = [
        'ListingKey', 'ListingKeyNumeric', 'ListingId', 'UnparsedAddress', 
        'ListPrice', 'OriginalListPrice', 'DaysOnMarket', 'AssociationFee', 
        'AssociationFeeFrequency', 'HighSchoolDistrict', 'BuyerAgentAOR', 
        'ListAgentAOR', 'BuyerOfficeAOR', 'ListAgentEmail', 'ListAgentFirstName', 
        'ListAgentLastName', 'ListAgentFullName', 'BuyerAgentFirstName', 
        'BuyerAgentLastName', 'BuyerAgentMlsId', 'ListOfficeName', 
        'BuyerOfficeName', 'SubdivisionName', 'Levels', 'Flooring', 
        'MLSAreaMajor', 'StreetNumberNumeric', 'MainLevelBedrooms', 'CoListOfficeName'
    ]
    cols_to_drop = [c for c in more_columns_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Step 5: Trim ClosePrice outliers (top and bottom 0.5%)
    if 'ClosePrice' in df.columns:
        close_price_bottom = df['ClosePrice'].quantile(0.005)
        close_price_top = df['ClosePrice'].quantile(0.995)
        df = df[(df['ClosePrice'] >= close_price_bottom) & (df['ClosePrice'] <= close_price_top)]
        df = df.dropna(subset=['ClosePrice'])  # Remove any remaining ClosePrice NaN
    
    # Step 6: Convert date columns to datetime
    date_columns = ['PurchaseContractDate', 'ListingContractDate', 'ContractStatusChangeDate', 'CloseDate']
    date_columns = [c for c in date_columns if c in df.columns]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Step 7: Fill Stories with 1 (default for single story)
    if 'Stories' in df.columns:
        df['Stories'] = df['Stories'].fillna(1)
    
    # Step 8: Fill GarageSpaces with ParkingTotal if missing
    if 'GarageSpaces' in df.columns and 'ParkingTotal' in df.columns:
        df.loc[df['GarageSpaces'].isna(), 'GarageSpaces'] = df.loc[df['GarageSpaces'].isna(), 'ParkingTotal']
    
    # Step 9: Handle LivingArea - drop if <1% missing, else fill with mean
    if 'LivingArea' in df.columns:
        if df['LivingArea'].isna().mean() < 0.01:
            df = df.dropna(subset=['LivingArea'])
        else:
            df['LivingArea'] = df['LivingArea'].fillna(df['LivingArea'].mean())
    
    # Step 10: Fill YearBuilt with median, then calculate PropertyAgeAtClose
    if 'YearBuilt' in df.columns and 'CloseDate' in df.columns:
        df['YearBuilt'] = df['YearBuilt'].fillna(df['YearBuilt'].median())
        if df['CloseDate'].dtype == 'datetime64[ns]':
            df['PropertyAgeAtClose'] = df['CloseDate'].dt.year - df['YearBuilt']
    
    # Step 11: Fill City with CountyOrParish if missing
    if 'City' in df.columns and 'CountyOrParish' in df.columns:
        df['City'] = df['City'].fillna(df['CountyOrParish'])
    
    # Step 12: Fill ParkingTotal with 0
    if 'ParkingTotal' in df.columns:
        df['ParkingTotal'] = df['ParkingTotal'].fillna(0)
    
    # Step 13: Fill Latitude/Longitude with mean of PostalCode
    if all(col in df.columns for col in ['Latitude', 'Longitude', 'PostalCode']):
        postal_means = (
            df[['PostalCode', 'Latitude', 'Longitude']]
            .dropna(subset=['PostalCode'])
            .groupby('PostalCode')
            .agg(mean_lat=('Latitude', 'mean'), mean_lon=('Longitude', 'mean'))
        )
        df['Latitude'] = df['Latitude'].fillna(
            df['PostalCode'].map(postal_means['mean_lat'])
        )
        df['Longitude'] = df['Longitude'].fillna(
            df['PostalCode'].map(postal_means['mean_lon'])
        )
    
    # Step 14: Convert boolean YN columns to bool and fill missing with False
    boolean_columns = ['ViewYN', 'PoolPrivateYN', 'AttachedGarageYN', 'FireplaceYN', 'NewConstructionYN']
    true_vals = {'true', 't', 'y', 'yes', '1'}
    
    for col in boolean_columns:
        if col in df.columns:
            if df[col].dtype == bool:
                # Already boolean, fill NaN with False
                df[col] = df[col].fillna(False)
            else:
                # Convert to boolean: True only for explicit True values, everything else False
                # Handle numeric True/False
                mask_true = (df[col] == True) | (df[col] == 1) | (df[col] == 'True') | (df[col] == 'true')
                # Handle string True values
                if df[col].dtype == object:
                    s = df[col].astype(str).str.strip().str.lower()
                    mask_true = mask_true | s.isin(true_vals)
                # Set: True where mask_true, False for everything else (including NaN)
                df[col] = mask_true.fillna(False).astype(bool)
    
    # Step 15: Fill any remaining numeric columns with median (if very few missing)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0 and df[col].isna().mean() < 0.05:  # <5% missing
            df[col] = df[col].fillna(df[col].median())
    
    # Step 16: Drop any remaining rows with NaN (final clean)
    df = df.dropna()
    
    return df