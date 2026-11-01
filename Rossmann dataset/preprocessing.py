"""Load, engineer and encode Rossmann store sales features."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_data(data_dir: str | Path):
    """Return training, 2015 validation and unlabeled test arrays."""
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    store_df = pd.read_csv(data_dir / "store.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    train_df = train_df.merge(store_df, on="Store", how="left")
    test_df = test_df.merge(store_df, on="Store", how="left")

    # The original experiment models sales only for open stores.
    train_df = train_df.loc[train_df["Open"] == 1].drop(
        columns=["Customers", "Open"]
    ).copy()
    test_df = test_df.drop(columns=["Open"])

    for frame in (train_df, test_df):
        dates = pd.to_datetime(frame["Date"])
        frame["Year"] = dates.dt.year
        frame["Month"] = dates.dt.month
        frame["Day"] = dates.dt.day
        frame["WeekOfYear"] = dates.dt.isocalendar().week.astype(int)
        frame.drop(columns=["Date"], inplace=True)

    median_distance = train_df["CompetitionDistance"].median()
    for frame in (train_df, test_df):
        frame["CompetitionDistance"] = frame["CompetitionDistance"].fillna(
            median_distance
        )
        for column in (
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2SinceWeek",
            "Promo2SinceYear",
        ):
            frame[column] = frame[column].fillna(0)

    # Retain the original chronological validation design.
    val_df = train_df.loc[train_df["Year"] == 2015].copy()
    train_df = train_df.loc[train_df["Year"] != 2015].copy()

    categorical_cols = ["StoreType", "Assortment", "StateHoliday"]
    numeric_cols = [
        "Store", "DayOfWeek", "Promo", "CompetitionDistance",
        "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2", "Promo2SinceWeek", "Promo2SinceYear",
        "SchoolHoliday", "Year", "Month", "Day", "WeekOfYear",
    ]
    for frame in (train_df, val_df, test_df):
        frame["StateHoliday"] = frame["StateHoliday"].astype(str)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    train_cat = encoder.fit_transform(train_df[categorical_cols])
    val_cat = encoder.transform(val_df[categorical_cols])
    test_cat = encoder.transform(test_df[categorical_cols])

    scaler = StandardScaler()
    train_num = scaler.fit_transform(train_df[numeric_cols])
    val_num = scaler.transform(val_df[numeric_cols])
    test_num = scaler.transform(test_df[numeric_cols])

    X_train = np.hstack([train_cat, train_num])
    X_val = np.hstack([val_cat, val_num])
    X_test = np.hstack([test_cat, test_num])

    y_train = train_df["Sales"].to_numpy()
    y_val = val_df["Sales"].to_numpy()
    return X_train, X_val, X_test, y_train, y_val
