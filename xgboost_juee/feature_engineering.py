"""
feature_engineering.py

Final, leakage-safe feature engineering for CA Single Family Residences.
Use engineer_features(train_df, test_df) to get processed train/test dataframes.

Assumptions:
- train_df contains ClosePrice (target).
- test_df may or may not contain ClosePrice (code tolerates missing ClosePrice in test).
- Date columns: CloseDate, ListingContractDate, PurchaseContractDate (datetime64[ns]).
- Latitude/Longitude exist and are numeric.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer
from math import radians, sin, cos, asin, sqrt

# -----------------------------
# Constants: COAST ANCHOR POINTS
# -----------------------------
COAST_POINTS = [
    (32.543, -117.124),  # San Diego
    (33.600, -117.900),  # Laguna Beach / Orange County
    (34.420, -119.698),  # Santa Barbara
    (34.281, -119.300),  # Ventura
    (34.014, -118.496),  # Santa Monica
    (36.600, -121.900),  # Monterey
    (36.778, -122.000),  # Santa Cruz
    (37.774, -122.510),  # San Francisco
    (38.300, -123.050),  # Sonoma Coast
    (40.440, -124.410),  # Humboldt Coast
]

# -----------------------------
# Helpers
# -----------------------------
def haversine_array(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine distance in kilometers between arrays lat1/lon1 and scalars lat2/lon2
    or between two arrays of the same shape.
    """
    # Convert all to radians
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0
    return R * c


def min_distance_to_coast(df, coast_points=COAST_POINTS):
    """
    For each row in df compute the minimum haversine distance to any coast anchor point.
    Returns a numpy array of distances (km).
    """
    lat = df["Latitude"].values
    lon = df["Longitude"].values
    distances = []
    for (clat, clon) in coast_points:
        distances.append(haversine_array(lat, lon, clat, clon))
    # shape: (n_points, n_rows) -> stack to (n_rows, n_points)
    stacked = np.vstack(distances).T
    return np.min(stacked, axis=1)


def kfold_target_encode(train_series, target, test_series, n_splits=5, random_state=42):
    """
    Leakage-safe KFold target encoding for a categorical series.
    Returns:
        - train_encoded (pd.Series aligned with train_series.index)
        - test_encoded (pd.Series with length len(test_series))
    Notes:
    - target must be aligned with train_series (same index/order).
    """
    train_series = train_series.reset_index(drop=True)
    target = target.reset_index(drop=True)
    test_series = test_series.reset_index(drop=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = pd.Series(index=train_series.index, dtype=float)
    test_mean_accum = np.zeros(len(test_series), dtype=float)

    for train_idx, valid_idx in kf.split(train_series):
        tr = train_series.iloc[train_idx]
        val = train_series.iloc[valid_idx]
        tr_target = target.iloc[train_idx]

        means = tr_target.groupby(tr).mean()
        oof.iloc[valid_idx] = val.map(means)

        # apply mapping to test and accumulate
        mapped_test = test_series.map(means).fillna(tr_target.mean()).values
        test_mean_accum += mapped_test

    test_mean = test_mean_accum / n_splits
    # fill any remaining NaN in oof with overall mean
    overall_mean = target.mean()
    oof.fillna(overall_mean, inplace=True)

    return oof.reset_index(drop=True), pd.Series(test_mean)


def safe_divide(a, b, fill=np.nan):
    """Elementwise safe division, replace div by zero with fill."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    out = np.full_like(a, fill, dtype=float)
    mask = (b != 0) & ~np.isnan(b)
    out[mask] = a[mask] / b[mask]
    return out


# -----------------------------
# Core feature adders
# -----------------------------
def add_coordinate_transformations(df):
    df = df.copy()
    df["Lat_Lon_Sum"] = df["Latitude"] + df["Longitude"]
    df["Lat_Lon_Diff"] = df["Latitude"] - df["Longitude"]
    df["Lat_Lon_Product"] = df["Latitude"] * df["Longitude"]
    df["LatLonMagnitude"] = np.sqrt(df["Latitude"]**2 + df["Longitude"]**2)
    return df


def add_size_efficiency_metrics(df, allow_price_per_sqft=False):
    df = df.copy()
    # Only compute PricePerSqFt if ClosePrice exists and allow_price_per_sqft True (train)
    if allow_price_per_sqft and "ClosePrice" in df.columns:
        df["PricePerSqFt"] = df["ClosePrice"] / df["LivingArea"].replace(0, np.nan)
    else:
        df["PricePerSqFt"] = np.nan  # placeholder; will be filled by mapping location benchmarks
    df["LotSizeToLivingArea"] = safe_divide(df["LotSizeSquareFeet"], df["LivingArea"])
    df["BathsPerBedroom"] = safe_divide(df["BathroomsTotalInteger"], df["BedroomsTotal"])
    df["BedsPerSqFt"] = safe_divide(df["BedroomsTotal"], df["LivingArea"])
    df["RoomsPerSqFt"] = safe_divide(df["BedroomsTotal"] + df["BathroomsTotalInteger"], df["LivingArea"])
    return df


def add_parking_usability(df):
    df = df.copy()
    df["GarageToParkingRatio"] = safe_divide(df["GarageSpaces"], df["ParkingTotal"])
    df["ParkingPerSqFt"] = safe_divide(df["ParkingTotal"], df["LivingArea"])
    df["GarageSpacesPerBedroom"] = safe_divide(df["GarageSpaces"], df["BedroomsTotal"])
    df["ParkingPerBedroom"] = safe_divide(df["ParkingTotal"], df["BedroomsTotal"])
    return df


def add_boolean_expansions(df):
    df = df.copy()
    # Convert booleans to ints if necessary
    bool_cols = ["ViewYN", "PoolPrivateYN", "AttachedGarageYN", "FireplaceYN", "NewConstructionYN"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)

    # Interaction flags
    if set(["PoolPrivateYN", "ViewYN"]).issubset(df.columns):
        df["HasPoolWithView"] = (df["PoolPrivateYN"] & df["ViewYN"]).astype(int)
    if set(["FireplaceYN", "AttachedGarageYN"]).issubset(df.columns):
        df["HasGarageAndFireplace"] = (df["FireplaceYN"] & df["AttachedGarageYN"]).astype(int)

    # Amenity score
    df["LuxuryAmenityScore"] = df[[c for c in bool_cols if c in df.columns]].sum(axis=1)
    return df


def add_close_date_features(df):
    df = df.copy()
    # Ensure datetime
    for date_col in ["CloseDate", "ListingContractDate", "PurchaseContractDate", "ContractStatusChangeDate"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if "CloseDate" in df.columns:
        df["CloseYear"] = df["CloseDate"].dt.year
        df["CloseMonth"] = df["CloseDate"].dt.month
        df["CloseQuarter"] = df["CloseDate"].dt.quarter
        df["CloseDayOfWeek"] = df["CloseDate"].dt.weekday
        df["CloseWeekOfYear"] = df["CloseDate"].dt.isocalendar().week.astype(int)
        df["CloseSeason"] = df["CloseMonth"].map({
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall", 10: "Fall", 11: "Fall"
        })
    # Duration features (no leakage: uses listing/purchase dates that are available pre-close)
    if set(["CloseDate", "ListingContractDate"]).issubset(df.columns):
        df["DaysOnMarket"] = (df["CloseDate"] - df["ListingContractDate"]).dt.days
    if set(["CloseDate", "PurchaseContractDate"]).issubset(df.columns):
        df["TimeFromOfferToClose"] = (df["CloseDate"] - df["PurchaseContractDate"]).dt.days
    if set(["PurchaseContractDate", "ListingContractDate"]).issubset(df.columns):
        df["TimeFromListingToOffer"] = (df["PurchaseContractDate"] - df["ListingContractDate"]).dt.days
    return df


def add_construction_features(df):
    df = df.copy()
    # Age at close (if CloseYear present)
    if "PropertyAgeAtClose" not in df.columns and "CloseYear" in df.columns and "YearBuilt" in df.columns:
        df["PropertyAgeAtClose"] = df["CloseYear"] - df["YearBuilt"]
    # Create age buckets
    if "PropertyAgeAtClose" in df.columns:
        df["AgeBucket"] = pd.cut(df["PropertyAgeAtClose"], bins=[-1, 5, 20, 50, 1000], labels=["0-5", "5-20", "20-50", "50+"])
        df["IsNewConstruction"] = (df["PropertyAgeAtClose"] <= 1).astype(int)
    # Era bins (example)
    if "YearBuilt" in df.columns:
        df["EraBin"] = pd.cut(df["YearBuilt"], bins=[0, 1950, 1980, 2000, 2010, 2025], labels=["<1950","1950-79","1980-99","2000-09","2010+"])
    return df


# -----------------------------
# Market velocity (leakage-safe)
# -----------------------------
def compute_market_velocity_train(train_df, location_col="PostalCode", window_days=90):
    """
    For training set: for each row, count how many train transactions in the same location
    occurred within [CloseDate - window_days, CloseDate) (i.e., prior window).
    Returns a pd.Series aligned with train_df index.
    """
    # Prepare
    df = train_df[[location_col, "CloseDate"]].copy()
    df = df.sort_values([location_col, "CloseDate"]).reset_index(drop=False)  # keep original index in 'index'
    orig_index = df["index"].values
    groups = df.groupby(location_col)

    result = np.zeros(len(df), dtype=int)

    for name, grp in groups:
        dates = grp["CloseDate"].values.astype("datetime64[ns]")
        # For each date, find number of dates >= date - window and < date
        # Use searchsorted on sorted array of dates
        left_bounds = dates - np.timedelta64(window_days, "D")
        # For each position i: count = i - left_idx
        positions = np.arange(len(dates))
        left_indices = np.searchsorted(dates, left_bounds, side="left")
        counts = positions - left_indices
        result[grp.index.values] = counts

    # Map back to original train index
    out_series = pd.Series(index=df["index"].values, data=result)
    out_series = out_series.sort_index().reindex(range(len(train_df)))  # ensure alignment
    out_series.index = train_df.index
    return out_series.astype(int)


def compute_market_velocity_test(train_df, test_df, location_col="PostalCode", window_days=90):
    """
    For test_df: for each test row, count number of train transactions in same location
    with CloseDate in [test_close - window_days, test_close).
    This uses TRAIN data only (no leakage).
    """
    # Build per-location sorted date arrays from train
    train_dates_by_loc = train_df.groupby(location_col)["CloseDate"].apply(lambda s: np.sort(s.values.astype("datetime64[ns]")))
    # For test rows, map location to date array and use searchsorted
    counts = []
    for idx, row in test_df.iterrows():
        loc = row.get(location_col, None)
        test_close = row.get("CloseDate", None)
        if pd.isna(loc) or pd.isna(test_close) or loc not in train_dates_by_loc.index:
            counts.append(0)
            continue
        arr = train_dates_by_loc.loc[loc]
        # window: [test_close - window_days, test_close)
        left = np.datetime64(test_close) - np.timedelta64(window_days, "D")
        # number of train dates < test_close
        right_idx = np.searchsorted(arr, np.datetime64(test_close), side="left")
        left_idx = np.searchsorted(arr, left, side="left")
        counts.append(int(max(0, right_idx - left_idx)))
    return pd.Series(counts, index=test_df.index, dtype=int)


# -----------------------------
# Location-level medians & PPSF (leakage-safe)
# -----------------------------
def compute_location_medians_and_ppsf(train_df, test_df, location_cols):
    train = train_df.copy()
    test = test_df.copy()

    # Temporary PPSF (for calculating medians)
    train["_PPSF_tmp"] = train["ClosePrice"] / train["LivingArea"].replace(0, np.nan)
    test["_PPSF_tmp"] = np.nan  # test doesn't have ClosePrice

    for col in location_cols:
        # ---- compute medians from TRAIN ONLY ----
        median_price = train.groupby(col)["ClosePrice"].median()
        median_ppsf = train.groupby(col)["_PPSF_tmp"].median()

        # assign medians back
        train[col + "_MedianPrice"] = train[col].map(median_price)
        test[col + "_MedianPrice"] = test[col].map(median_price)

        train[col + "_MedianPPSF"] = train[col].map(median_ppsf)
        test[col + "_MedianPPSF"] = test[col].map(median_ppsf)

        # ---- TRAIN ONLY: relative features (test has no ClosePrice) ----
        train[col + "_RelPriceToMedian"] = train["ClosePrice"] / train[col + "_MedianPrice"]
        test[col + "_RelPriceToMedian"] = np.nan  # avoid leakage

        train[col + "_RelPPSFToMedian"] = train["_PPSF_tmp"] / train[col + "_MedianPPSF"]
        test[col + "_RelPPSFToMedian"] = np.nan

    # Fill missing medians in test using global medians
    global_price_med = train["ClosePrice"].median()
    global_ppsf_med = train["_PPSF_tmp"].median()

    for col in location_cols:
        test[col + "_MedianPrice"].fillna(global_price_med, inplace=True)
        test[col + "_MedianPPSF"].fillna(global_ppsf_med, inplace=True)

    train.drop(columns="_PPSF_tmp", inplace=True)
    test.drop(columns="_PPSF_tmp", inplace=True)

    return train, test



# -----------------------------
# KMeans Geo Clusters (no leakage: use both train+test coords only)
# -----------------------------
def add_geo_clusters(train_df, test_df, n_clusters=20, random_state=42):
    combo = pd.concat([
        train_df[["Latitude", "Longitude"]],
        test_df[["Latitude", "Longitude"]]
    ], axis=0, ignore_index=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(combo.values)
    train_clusters = clusters[:len(train_df)]
    test_clusters = clusters[len(train_df):]
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["GeoCluster"] = train_clusters
    test_df["GeoCluster"] = test_clusters
    return train_df, test_df


# -----------------------------
# Outlier-resistant transforms (fit on train only)
# -----------------------------
def fit_power_transformer(train_df, cols):
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    pt.fit(train_df[cols].fillna(0).values)
    return pt


def apply_power_transformer(df, cols, pt):
    arr = pt.transform(df[cols].fillna(0).values)
    out = pd.DataFrame(arr, columns=[c + "_YJ" for c in cols], index=df.index)
    return pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)


# -----------------------------
# Top-level engineer_features
# -----------------------------
def engineer_features(train_df, test_df, coast_points=COAST_POINTS, kmeans_clusters=20):
    """
    Master pipeline that returns (train_fe, test_fe, metadata)
    metadata contains fitted objects used (power transformer, target encoders maps, kmeans if needed)
    """

    # Make copies
    train = train_df.copy().reset_index(drop=True)
    test = test_df.copy().reset_index(drop=True)

    # Ensure date columns as datetime
    for col in ["CloseDate", "ListingContractDate", "PurchaseContractDate", "ContractStatusChangeDate"]:
        if col in train.columns:
            train[col] = pd.to_datetime(train[col], errors="coerce")
        if col in test.columns:
            test[col] = pd.to_datetime(test[col], errors="coerce")

    # -----------------------
    # 1) Coordinate transforms
    # -----------------------
    train = add_coordinate_transformations(train)
    test = add_coordinate_transformations(test)

    # -----------------------
    # 2) Distance to coast (min of anchors)
    # -----------------------
    train["DistToCoast_km"] = min_distance_to_coast(train, coast_points)
    test["DistToCoast_km"] = min_distance_to_coast(test, coast_points)

    # -----------------------
    # 3) Geo clusters (kmeans on combined coords OK)
    # -----------------------
    train, test = add_geo_clusters(train, test, n_clusters=kmeans_clusters)

    # -----------------------
    # 4) Close date features
    # -----------------------
    train = add_close_date_features(train)
    test = add_close_date_features(test)

    # -----------------------
    # 5) Construction features
    # -----------------------
    train = add_construction_features(train)
    test = add_construction_features(test)

    # -----------------------
    # 6) Size efficiency & parking
    # For training we can compute PricePerSqFt; for test we will rely on location benchmarks
    # -----------------------
    train = add_size_efficiency_metrics(train, allow_price_per_sqft=True)
    test = add_size_efficiency_metrics(test, allow_price_per_sqft=False)

    train = add_parking_usability(train)
    test = add_parking_usability(test)

    # -----------------------
    # 7) Boolean expansions
    # -----------------------
    train = add_boolean_expansions(train)
    test = add_boolean_expansions(test)

    # -----------------------
    # 8) Market velocity (leakage-safe)
    # -----------------------
    # Train: count of prior window_days (default 90). For test: counts of TRAIN transactions in prior window.
    window_days = 90
    # Ensure CloseDate exists
    if "PostalCode" in train.columns:
        train["MarketVelocity_90d"] = compute_market_velocity_train(train, location_col="PostalCode", window_days=window_days)
        test["MarketVelocity_90d"] = compute_market_velocity_test(train, test, location_col="PostalCode", window_days=window_days)
    else:
        # fallback to city
        train["MarketVelocity_90d"] = compute_market_velocity_train(train, location_col="City", window_days=window_days)
        test["MarketVelocity_90d"] = compute_market_velocity_test(train, test, location_col="City", window_days=window_days)

    # -----------------------
    # 9) Location medians & PPSF (train only)
    # -----------------------
    loc_cols = ["PostalCode", "City", "CountyOrParish"]
    train, test = compute_location_medians_and_ppsf(train, test, location_cols=loc_cols)

    # Fill PricePerSqFt in test using location median PPSF
    # If test.PricePerSqFt is NaN, fill with PostalCode median PPSF, else City, else County, else global
    global_ppsf = train["ClosePrice"].div(train["LivingArea"].replace(0, np.nan)).median()
    def fill_ppsf_row(r):
        if not np.isnan(r.get("PricePerSqFt", np.nan)):
            return r["PricePerSqFt"]
        # try PostalCode
        if "PostalCode_MedianPPSF" in r and not pd.isna(r["PostalCode_MedianPPSF"]):
            return r["PostalCode_MedianPPSF"]
        if "City_MedianPPSF" in r and not pd.isna(r["City_MedianPPSF"]):
            return r["City_MedianPPSF"]
        if "CountyOrParish_MedianPPSF" in r and not pd.isna(r["CountyOrParish_MedianPPSF"]):
            return r["CountyOrParish_MedianPPSF"]
        return global_ppsf

    test["PricePerSqFt"] = test.apply(fill_ppsf_row, axis=1)

    # -----------------------
    # 10) Target encoding (KFold) for City, PostalCode, County (train-only KFold)
    # -----------------------
    # We will create TE for City, PostalCode, County using train ClosePrice
    te_metadata = {}
    for col in ["City", "PostalCode", "CountyOrParish"]:
        if col not in train.columns or train[col].isna().all():
            train[col + "_TE"] = np.nan
            test[col + "_TE"] = np.nan
            te_metadata[col] = None
            continue
        tr_enc, te_enc = kfold_target_encode(train[col], train["ClosePrice"], test[col], n_splits=5)
        train[col + "_TE"] = tr_enc.values
        test[col + "_TE"] = te_enc.values
        te_metadata[col] = {
            "train_mapping": train[[col, col + "_TE"]].groupby(col)[col + "_TE"].first().to_dict()
        }

    # -----------------------
    # 11) Outlier-resistant transforms (Yeo-Johnson), fit on train only
    # -----------------------
    pt_cols = ["LivingArea", "LotSizeSquareFeet", "ClosePrice"] if "ClosePrice" in train.columns else ["LivingArea", "LotSizeSquareFeet"]
    pt = fit_power_transformer(train, pt_cols)
    train = apply_power_transformer(train, pt_cols, pt)
    test = apply_power_transformer(test, pt_cols, pt)

    # -----------------------
    # 12) Coordinate / size interactions & extras
    # -----------------------
    # Example interactions
    for df in [train, test]:
        df["Lat_Long_Interaction"] = df["Latitude"] * df["Longitude"]
        df["LivingArea_per_Story"] = safe_divide(df["LivingArea"], df["Stories"].replace(0, np.nan))
        df["LotSize_per_Story"] = safe_divide(df["LotSizeSquareFeet"], df["Stories"].replace(0, np.nan))

    # -----------------------
    # 13) Final clean-up: ensure consistent dtypes
    # -----------------------
    # Convert boolean-like ints to int dtype
    bool_like = ["ViewYN", "PoolPrivateYN", "AttachedGarageYN", "FireplaceYN", "NewConstructionYN",
                 "HasPoolWithView", "HasGarageAndFireplace", "IsNewConstruction"]
    for c in bool_like:
        if c in train.columns:
            train[c] = train[c].fillna(0).astype(int)
        if c in test.columns:
            test[c] = test[c].fillna(0).astype(int)

    # -----------------------
    # metadata to return for saving / later use
    # -----------------------
    metadata = {
        "coast_points": coast_points,
        "kmeans_clusters": kmeans_clusters,
        "power_transformer": pt,
        "target_encoding_meta": te_metadata,
    }

    return train, test, metadata

