import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans


# -------------------------
# Helper Functions
# -------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorized Haversine formula."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def kfold_target_encode(train_series, target, test_series, n_splits=5):
    """Leakage-safe KFold target encoding."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = pd.Series(np.nan, index=train_series.index)
    test_encoded = pd.Series(np.zeros(len(test_series)))

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
# NEW: Coastal Anchor Points for California
# --------------------------------------------------------

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


def min_distance_to_coast(df, coast_points):
    """Compute minimum distance to any major CA coastal anchor point."""
    all_distances = []
    for lat_c, lon_c in coast_points:
        dist = haversine_distance(df["Latitude"], df["Longitude"], lat_c, lon_c)
        all_distances.append(dist)
    return np.column_stack(all_distances).min(axis=1)


# --------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------

def build_features(train_cleaned, test_clean):

    # Combine for KMeans clustering
    full = pd.concat([train_cleaned.drop(columns=["ClosePrice"]),
                      test_clean],
                     axis=0,
                     ignore_index=True)

    # --------------------------------------------------------
    # 1. DATE FEATURES
    # --------------------------------------------------------
    for df in [train_cleaned, test_clean]:
        df["CloseYear"] = df["CloseDate"].dt.year
        df["CloseMonth"] = df["CloseDate"].dt.month
        df["CloseQuarter"] = df["CloseDate"].dt.quarter

        df["CloseSeason"] = df["CloseMonth"].map({
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall", 10: "Fall", 11: "Fall"
        })

        df["DaysOnMarket"] = (df["CloseDate"] - df["ListingContractDate"]).dt.days
        df["TimeFromOfferToClose"] = (df["CloseDate"] - df["PurchaseContractDate"]).dt.days
        df["TimeFromListingToOffer"] = (df["PurchaseContractDate"] - df["ListingContractDate"]).dt.days

    # --------------------------------------------------------
    # 2. LOG TRANSFORMS & BUCKETS
    # --------------------------------------------------------
    for df in [train_cleaned, test_clean]:
        df["LogLivingArea"] = np.log1p(df["LivingArea"])
        df["LogLotSizeSF"] = np.log1p(df["LotSizeSquareFeet"])

        df["AgeBucket"] = pd.cut(df["PropertyAgeAtClose"],
                                 bins=[-1, 5, 20, 50, 200],
                                 labels=["0-5", "5-20", "20-50", "50+"])

    # --------------------------------------------------------
    # 3. RATIO FEATURES
    # --------------------------------------------------------
    def add_ratios(df):
        df["LotSizeToLivingArea"] = df["LotSizeSquareFeet"] / df["LivingArea"]
        df["BathsPerSqFt"] = df["BathroomsTotalInteger"] / df["LivingArea"]
        df["BedsPerSqFt"] = df["BedroomsTotal"] / df["LivingArea"]
        df["RoomsPerSqFt"] = (df["BedroomsTotal"] + df["BathroomsTotalInteger"]) / df["LivingArea"]
        df["ParkingPerSqFt"] = df["ParkingTotal"] / df["LivingArea"]

        df["GarageSpacesPerBedroom"] = df["GarageSpaces"] / df["BedroomsTotal"]
        df["Lat_Long_Interaction"] = df["Latitude"] * df["Longitude"]

        df["IsSingleStory"] = (df["Stories"] == 1).astype(int)
        df["IsTwoStory"] = (df["Stories"] == 2).astype(int)

    add_ratios(train_cleaned)
    add_ratios(test_clean)

    # --------------------------------------------------------
    # 4. INTERACTION BOOLEAN FEATURES
    # --------------------------------------------------------
    for df in [train_cleaned, test_clean]:
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
    # 5. UPDATED: MULTI-POINT DISTANCE TO COAST
    # --------------------------------------------------------
    train_cleaned["DistToCoast_km"] = min_distance_to_coast(train_cleaned, COAST_POINTS)
    test_clean["DistToCoast_km"] = min_distance_to_coast(test_clean, COAST_POINTS)

    # --------------------------------------------------------
    # 6. KMeans GEO CLUSTERING
    # --------------------------------------------------------
    coords = full[["Latitude", "Longitude"]]
    km = KMeans(n_clusters=20, random_state=42, n_init=10)
    full["GeoCluster"] = km.fit_predict(coords)

    train_cleaned["GeoCluster"] = full.iloc[: len(train_cleaned)]["GeoCluster"].values
    test_clean["GeoCluster"] = full.iloc[len(train_cleaned):]["GeoCluster"].values

    # --------------------------------------------------------
    # 7. TARGET ENCODING (City, Zip, County)
    # --------------------------------------------------------
    train_y = train_cleaned["ClosePrice"]

    for col in ["City", "PostalCode", "CountyOrParish"]:
        tr_enc, te_enc = kfold_target_encode(train_cleaned[col], train_y, test_clean[col])
        train_cleaned[f"{col}_TE"] = tr_enc
        test_clean[f"{col}_TE"] = te_enc

    return train_cleaned, test_clean

