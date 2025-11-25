import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from typing import Tuple, Optional, Dict

def adaptive_clean_real_estate_data(
    df: pd.DataFrame,
    imputation_stats: Optional[Dict] = None,
    missing_mechanisms: Optional[Dict] = None,
    fit: bool = False,
    trim_outliers: bool = False,
    use_model_imputation: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Optional[Dict], Optional[Dict]]:
    """
    Clean real estate data with adaptive imputation based on missing data mechanisms.
    
    This function diagnoses WHY data is missing (MCAR, MAR, MNAR) and applies
    appropriate imputation strategies to avoid bias.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw real estate dataset
    imputation_stats : dict, optional
        Pre-computed statistics (required when fit=False)
    missing_mechanisms : dict, optional
        Pre-diagnosed missing mechanisms (if already computed)
    fit : bool, default=False
        If True, diagnose mechanisms and compute stats (training set)
        If False, use provided stats (test set)
    trim_outliers : bool, default=False
        If True, remove top/bottom 0.5% of ClosePrice (test set only)
    use_model_imputation : bool, default=True
        If True, use IterativeImputer for MAR data. If False, use simple imputation
    verbose : bool, default=True
        Print diagnostic information
        
    Returns:
    --------
    df_clean : pd.DataFrame
        Cleaned dataset
    imputation_stats : dict or None
        Imputation statistics (only if fit=True)
    missing_mechanisms : dict or None
        Diagnosed missing mechanisms (only if fit=True)
    """
    
    if not fit and imputation_stats is None:
        raise ValueError("imputation_stats required when fit=False")
    
    initial_rows = len(df)
    df = df.copy()
    
    if verbose:
        print("="*80)
        print(f"ADAPTIVE CLEANING - {'TRAINING' if fit else 'TEST'} DATA")
        print("="*80)
        print(f"Initial shape: {df.shape}\n")
    
    # ========================================================================
    # PHASE 1: Basic cleaning (same as before)
    # ========================================================================
    
    # Drop high-missing columns
    threshold = 0.75
    cols_to_drop = df.columns[df.isna().mean() > threshold].tolist()
    if verbose and cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns with >75% missing")
    df = df.drop(columns=cols_to_drop)
    
    # Apply required filters
    if 'StateOrProvince' in df.columns:
        df = df[df['StateOrProvince'] == 'CA']
    if 'PropertyType' in df.columns:
        df = df[df['PropertyType'] == 'Residential']
    if 'PropertySubType' in df.columns:
        df = df[df['PropertySubType'] == 'SingleFamilyResidence']
    
    # Drop zero-variance and excluded columns
    zero_variance_cols = df.columns[df.nunique() <= 1].tolist()
    df = df.drop(columns=zero_variance_cols)
    
    columns_to_exclude = [
        'ListingKey', 'ListingKeyNumeric', 'ListingId', 'UnparsedAddress',
        'ListPrice', 'OriginalListPrice', 'DaysOnMarket', 'AssociationFee',
        'AssociationFeeFrequency', 'HighSchoolDistrict', 'BuyerAgentAOR',
        'ListAgentAOR', 'BuyerOfficeAOR', 'ListAgentEmail', 'ListAgentFirstName',
        'ListAgentLastName', 'ListAgentFullName', 'BuyerAgentFirstName',
        'BuyerAgentLastName', 'BuyerAgentMlsId', 'ListOfficeName',
        'BuyerOfficeName', 'SubdivisionName', 'Levels', 'Flooring',
        'MLSAreaMajor', 'StreetNumberNumeric', 'MainLevelBedrooms',
        'StateOrProvince', 'PropertyType', 'PropertySubType', 'SourceFile', 'CoListOfficeName'
    ]
    cols_to_drop = [c for c in columns_to_exclude if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Handle ClosePrice
    if trim_outliers and 'ClosePrice' in df.columns:
        close_price_bottom = df['ClosePrice'].quantile(0.005)
        close_price_top = df['ClosePrice'].quantile(0.995)
        df = df[(df['ClosePrice'] >= close_price_bottom) & (df['ClosePrice'] <= close_price_top)]
    
    if 'ClosePrice' in df.columns:
        df = df.dropna(subset=['ClosePrice'])
    
    # Convert dates
    date_columns = ['PurchaseContractDate', 'ListingContractDate',
                   'ContractStatusChangeDate', 'CloseDate']
    date_columns = [c for c in date_columns if c in df.columns]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # ========================================================================
    # PHASE 2: Diagnose missing data mechanisms (TRAINING ONLY)
    # ========================================================================
    
    if fit:
        if verbose:
            print("\n" + "="*80)
            print("DIAGNOSING MISSING DATA MECHANISMS")
            print("="*80)
        
        imputation_stats = {}
        missing_mechanisms = {}
        
        # Analyze each column with missing data
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            
            missing_pct = df[col].isnull().mean()
            
            # Diagnose mechanism (simplified version)
            mechanism = 'MCAR'  # Default assumption
            
            # Check for MAR signals
            if df[col].dtype in [np.float64, np.int64]:
                # For numeric: check correlation with other features
                df[f'{col}_missing'] = df[col].isnull().astype(int)
                
                correlations = []
                for other_col in df.select_dtypes(include=[np.number]).columns:
                    if other_col != col and not other_col.endswith('_missing'):
                        corr = df[f'{col}_missing'].corr(df[other_col])
                        if abs(corr) > 0.1:  # Threshold for significant correlation
                            correlations.append((other_col, corr))
                
                if len(correlations) > 0:
                    mechanism = 'MAR'
                    if verbose:
                        print(f"\n{col}: MAR detected (correlates with {len(correlations)} features)")
                        for feat, corr in correlations[:3]:
                            print(f"  - {feat}: r={corr:.3f}")
                
                df = df.drop(columns=[f'{col}_missing'])
            
            # Domain-specific MNAR checks
            if col in ['PoolPrivateYN', 'ViewYN', 'FireplaceYN', 'AttachedGarageYN']:
                mechanism = 'NMAR'
                if verbose:
                    print(f"\n{col}: NMAR (amenity - missing likely means absent)")
            
            missing_mechanisms[col] = {
                'mechanism': mechanism,
                'missing_pct': missing_pct * 100
            }
        
        if verbose:
            print("\n" + "-"*80)
            print("MECHANISM SUMMARY:")
            mcar = sum(1 for m in missing_mechanisms.values() if m['mechanism'] == 'MCAR')
            mar = sum(1 for m in missing_mechanisms.values() if m['mechanism'] == 'MAR')
            nmar = sum(1 for m in missing_mechanisms.values() if m['mechanism'] == 'NMAR')
            print(f"  MCAR: {mcar} columns")
            print(f"  MAR:  {mar} columns")
            print(f"  NMAR: {nmar} columns")
    
    # ========================================================================
    # PHASE 3: Adaptive imputation based on mechanism
    # ========================================================================
    
    if verbose:
        print("\n" + "="*80)
        print("ADAPTIVE IMPUTATION")
        print("="*80)
    
    # Handle different mechanisms differently
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        
        mechanism = missing_mechanisms.get(col, {}).get('mechanism', 'MCAR') if missing_mechanisms else 'MCAR'
        
        if mechanism == 'NMAR':
            # For NMAR (amenities): Create indicator + fill with False/0
            if df[col].dtype == bool or col.endswith('YN'):
                if fit:
                    imputation_stats[f'{col}_missing_indicator'] = True
                # Create missing indicator
                df[f'{col}_was_missing'] = df[col].isnull().astype(int)
                # Fill with False (absent)
                df[col] = df[col].fillna(False)
                if verbose:
                    print(f"\n{col} (NMAR): Created missing indicator, filled with False")
            else:
                if fit:
                    imputation_stats[f'{col}_missing_indicator'] = True
                df[f'{col}_was_missing'] = df[col].isnull().astype(int)
                # Fill with 0 or appropriate default
                df[col] = df[col].fillna(0)
                if verbose:
                    print(f"\n{col} (NMAR): Created missing indicator, filled with 0")
        
        elif mechanism == 'MAR' and use_model_imputation:
            # For MAR: Use model-based imputation
            if df[col].dtype in [np.float64, np.int64]:
                if fit:
                    # We'll use IterativeImputer later for all MAR numeric features
                    imputation_stats[f'{col}_mar'] = True
                if verbose:
                    print(f"\n{col} (MAR): Will use model-based imputation")
        
        elif mechanism == 'MCAR':
            # For MCAR: Simple imputation is fine
            if df[col].dtype in [np.float64, np.int64]:
                if fit:
                    if col in ['Stories']:
                        imputation_stats[col] = 1.0
                    elif col in ['ParkingTotal', 'GarageSpaces']:
                        imputation_stats[col] = 0.0
                    else:
                        imputation_stats[col] = df[col].median()
                
                fill_value = imputation_stats.get(col, df[col].median())
                df[col] = df[col].fillna(fill_value)
                
                if verbose:
                    print(f"\n{col} (MCAR): Filled with {fill_value:.2f}")
    
    # ========================================================================
    # PHASE 4: Model-based imputation for MAR numeric features
    # ========================================================================
    
    if use_model_imputation:
        mar_numeric_cols = []
        if missing_mechanisms:
            mar_numeric_cols = [
                col for col, info in missing_mechanisms.items()
                if info['mechanism'] == 'MAR' and col in df.columns and df[col].dtype in [np.float64, np.int64]
            ]
        
        if len(mar_numeric_cols) > 0 and df[mar_numeric_cols].isnull().sum().sum() > 0:
            if verbose:
                print(f"\n" + "-"*80)
                print(f"Applying IterativeImputer to {len(mar_numeric_cols)} MAR numeric features...")
            
            # Select numeric features for imputation
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if not c.endswith('_was_missing')]
            
            if fit:
                imputer = IterativeImputer(max_iter=10, random_state=42)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                imputation_stats['iterative_imputer'] = imputer
                if verbose:
                    print("  ✓ Fitted IterativeImputer")
            else:
                if 'iterative_imputer' in imputation_stats:
                    df[numeric_cols] = imputation_stats['iterative_imputer'].transform(df[numeric_cols])
                    if verbose:
                        print("  ✓ Applied IterativeImputer")
    
    # ========================================================================
    # PHASE 5: Remaining cleaning steps
    # ========================================================================
    
    # PropertyAgeAtClose
    if 'YearBuilt' in df.columns and 'CloseDate' in df.columns:
        if df['CloseDate'].dtype == 'datetime64[ns]':
            df['PropertyAgeAtClose'] = df['CloseDate'].dt.year - df['YearBuilt']
    
    # Geographic coordinates (cascading imputation)
    if all(col in df.columns for col in ['Latitude', 'Longitude', 'PostalCode']):
        if fit:
            postal_means = df.groupby('PostalCode').agg(
                mean_lat=('Latitude', 'mean'),
                mean_lon=('Longitude', 'mean')
            )
            county_means = df.groupby('CountyOrParish').agg(
                mean_lat=('Latitude', 'mean'),
                mean_lon=('Longitude', 'mean')
            ) if 'CountyOrParish' in df.columns else None
            
            imputation_stats['postal_lat_means'] = postal_means['mean_lat'].to_dict()
            imputation_stats['postal_lon_means'] = postal_means['mean_lon'].to_dict()
            if county_means is not None:
                imputation_stats['county_lat_means'] = county_means['mean_lat'].to_dict()
                imputation_stats['county_lon_means'] = county_means['mean_lon'].to_dict()
            imputation_stats['overall_lat_mean'] = df['Latitude'].mean()
            imputation_stats['overall_lon_mean'] = df['Longitude'].mean()
        
        # Apply cascading imputation
        df['Latitude'] = df['Latitude'].fillna(df['PostalCode'].map(imputation_stats.get('postal_lat_means', {})))
        df['Longitude'] = df['Longitude'].fillna(df['PostalCode'].map(imputation_stats.get('postal_lon_means', {})))
        
        if 'CountyOrParish' in df.columns:
            df['Latitude'] = df['Latitude'].fillna(df['CountyOrParish'].map(imputation_stats.get('county_lat_means', {})))
            df['Longitude'] = df['Longitude'].fillna(df['CountyOrParish'].map(imputation_stats.get('county_lon_means', {})))
        
        df['Latitude'] = df['Latitude'].fillna(imputation_stats.get('overall_lat_mean'))
        df['Longitude'] = df['Longitude'].fillna(imputation_stats.get('overall_lon_mean'))
    
    # Boolean columns
    boolean_columns = ['ViewYN', 'PoolPrivateYN', 'AttachedGarageYN',
                      'FireplaceYN', 'NewConstructionYN']
    for col in boolean_columns:
        if col in df.columns:
            if df[col].dtype != bool:
                df[col] = df[col].astype(str).str.lower().str.strip()
                bool_map = {
                    'true': True, 't': True, 'y': True, 'yes': True,
                    '1': True, '1.0': True,
                    'false': False, 'f': False, 'n': False, 'no': False,
                    '0': False, '0.0': False, 'nan': False, 'none': False
                }
                df[col] = df[col].map(bool_map).fillna(False).astype(bool)
            else:
                df[col] = df[col].fillna(False)
    
    # Final cleanup
    remaining_nulls = df.isnull().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    
    if len(remaining_nulls) > 0:
        if verbose:
            print(f"\n⚠️  Remaining nulls: {remaining_nulls.to_dict()}")
        df = df.dropna()
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    if verbose:
        print("\n" + "="*80)
        print("CLEANING COMPLETE")
        print("="*80)
        print(f"Final shape: {df.shape}")
        print(f"Rows retained: {len(df)}/{initial_rows} ({len(df)/initial_rows*100:.1f}%)")
        
        # Show missingness indicator columns created
        missing_indicators = [c for c in df.columns if c.endswith('_was_missing')]
        if missing_indicators:
            print(f"\nMissingness indicators created: {len(missing_indicators)}")
            for col in missing_indicators:
                pct = df[col].mean() * 100
                print(f"  - {col}: {pct:.1f}% of data")
        
        print("="*80)
    
    if fit:
        return df, imputation_stats, missing_mechanisms
    else:
        return df, None, None

