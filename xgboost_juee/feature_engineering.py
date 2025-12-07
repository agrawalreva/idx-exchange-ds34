import pandas as pd
import numpy as np

def build_features(train_df, test_df):

    # -------------------------
    # 1. DATE FEATURES
    # -------------------------
    for df in [train_df, test_df]:
        df["CloseYear"] = df["CloseDate"].dt.year
        df["CloseMonth"] = df["CloseDate"].dt.month
        df["CloseQuarter"] = df["CloseDate"].dt.quarter
        df["CloseDayOfWeek"] = df["CloseDate"].dt.dayofweek

        df["DaysOnMarket"] = (df["CloseDate"] - df["ListingContractDate"]).dt.days
        df["DaysOfferToClose"] = (df["CloseDate"] - df["PurchaseContractDate"]).dt.days

    # -------------------------
    # 2. LOG FEATURES
    # -------------------------
    for df in [train_df, test_df]:
        df["LogLivingArea"] = np.log1p(df["LivingArea"])
        df["LogLotSize"] = np.log1p(df["LotSizeSquareFeet"])

    # -------------------------
    # 3. STRUCTURAL RATIOS
    # -------------------------

    # BOTH train + test get these:
    for df in [train_df, test_df]:
        df["PricePerSqFt"] = df["ClosePrice"] / (df["LivingArea"] + 1)
        df["BathsPerBedroom"] = df["BathroomsTotalInteger"] / (df["BedroomsTotal"] + 1)
        df["GarageToParking"] = (df["GarageSpaces"] / (df["ParkingTotal"] + 1)).replace([np.inf, -np.inf], 0)

    # -------------------------
    # 4. SIMPLE BOOLEAN INTERACTIONS
    # -------------------------
    for df in [train_df, test_df]:
        df["HasPoolWithView"] = (df["PoolPrivateYN"] & df["ViewYN"]).astype(int)
        df["IsSingleStory"] = (df["Stories"] == 1).astype(int)
        df["IsTwoStory"] = (df["Stories"] == 2).astype(int)

    # -------------------------
    # 5. DROP UNUSED COLUMNS
    # -------------------------
    drop_cols = [
        "PurchaseContractDate",
        "ListingContractDate",
        "CloseDate",
        "ContractStatusChangeDate",
        "LotSizeSquareFeet",
        "LotSizeArea",
        "City", "CountyOrParish", "PostalCode",
    ]

    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=drop_cols, errors="ignore")

    return train_df, test_df
