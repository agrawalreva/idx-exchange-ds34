import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans


# -------------------------
# Helper Functions
# -------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine formula
    """
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def kfold_target_encode(train_series, target, test_series, n_splits=5):
    """
    Leakage-safe KFold target encoding.
    """
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
# BEGIN MAIN PIPELINE
# --------------------------------------------------------

def build_features(train_cleaned, test_clean):

    # Combine temporarily (useful for KMeans, etc.)
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
    # 2. LOG TRANSFORM FEATURES
    # --------------------------------------------------------
    for df in [train_cleaned, test_clean]:
        df["LogLivingArea"] = np.log1p(df["LivingArea"])
        df["LogLotSizeSF"] = np.log1p(df["LotSizeSquareFeet"])

    # --------------------------------------------------------
    # 3. RATIO FEATURES
    # --------------------------------------------------------
    def add_ratios(df):
        df["LotSizeToLivingArea"] = df["LotSizeSquareFeet"] / (df["LivingArea"] + 1)
        df["BathsPerSqFt"] = df["BathroomsTotalInteger"] / (df["LivingArea"] + 1)
        df["BedsPerSqFt"] = df["BedroomsTotal"] / (df["LivingArea"] + 1)
        df["RoomsPerSqFt"] = (df["BedroomsTotal"] + df["BathroomsTotalInteger"]) / (df["LivingArea"] + 1)
        df["ParkingPerSqFt"] = df["ParkingTotal"] / (df["LivingArea"] + 1)
        df["GarageSpacesPerBedroom"] = df["GarageSpaces"] / (df["BedroomsTotal"] + 1)
        df["Lat_Long_Interaction"] = df["Latitude"] * df["Longitude"]

        df["IsSingleStory"] = (df["Stories"] == 1).astype(int)
        df["IsTwoStory"] = (df["Stories"] == 2).astype(int)

    add_ratios(train_cleaned)
    add_ratios(test_clean)

    # --------------------------------------------------------
    # 4. Interaction Boolean Features
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
    # 5. Distance to Coast
    # --------------------------------------------------------
    # Approximate reference: central CA coastline point
    coast_lat, coast_lon = 36.7783, -121.4179

    train_cleaned["DistToCoast_km"] = haversine_distance(
        train_cleaned["Latitude"], train_cleaned["Longitude"],
        coast_lat, coast_lon
    )
    test_clean["DistToCoast_km"] = haversine_distance(
        test_clean["Latitude"], test_clean["Longitude"],
        coast_lat, coast_lon
    )

    # --------------------------------------------------------
    # 6. KMeans Lat/Long Clusters
    # --------------------------------------------------------
    coords = full[["Latitude", "Longitude"]]
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    full["GeoCluster"] = kmeans.fit_predict(coords)

    train_cleaned["GeoCluster"] = full.iloc[: len(train_cleaned)]["GeoCluster"].values
    test_clean["GeoCluster"] = full.iloc[len(train_cleaned):]["GeoCluster"].values

    # --------------------------------------------------------
    # 7. Target Encoded City / Zip / County
    # --------------------------------------------------------
    train_y = train_cleaned["ClosePrice"]

    for col in ["City", "PostalCode", "CountyOrParish"]:
        train_te, test_te = kfold_target_encode(
            train_cleaned[col], train_y, test_clean[col]
        )
        train_cleaned[f"{col}_TE"] = train_te
        test_clean[f"{col}_TE"] = test_te

    # Return final
    train_cleaned = train_cleaned.drop(columns = ['PurchaseContractDate', 'ListingContractDate', 'CloseDate', 'ContractStatusChangeDate', 'PostalCode', 'CountyOrParish', 'City', 'LotSizeSquareFeet', 'LotSizeArea'])
    test_clean = test_clean.drop(columns = ['PurchaseContractDate', 'ListingContractDate', 'CloseDate', 'ContractStatusChangeDate', 'PostalCode', 'CountyOrParish', 'City', 'LotSizeSquareFeet', 'LotSizeArea'])
    return train_cleaned, test_clean


