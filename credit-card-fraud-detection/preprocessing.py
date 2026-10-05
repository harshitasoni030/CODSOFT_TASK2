"""Shared preprocessing: used by BOTH training and the Flask app,
so predictions always get exactly the same treatment as training data."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "is_fraud"

# Raw columns taken from the dataset (what the user enters in the form)
RAW_FEATURES = [
    "trans_date_trans_time", "amt", "category", "gender", "state",
    "city_pop", "lat", "long", "merch_lat", "merch_long", "dob",
]

# Columns the model actually sees after feature engineering
NUMERIC = ["amt", "city_pop", "hour", "day_of_week", "age", "distance_km"]
CATEGORICAL = ["category", "gender", "state"]


def add_features(df):
    """Turn raw columns into model features."""
    df = df.copy()
    for col in ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long"]:
        df[col] = pd.to_numeric(df[col])

    trans_time = pd.to_datetime(df["trans_date_trans_time"])
    dob = pd.to_datetime(df["dob"])
    df["hour"] = trans_time.dt.hour
    df["day_of_week"] = trans_time.dt.dayofweek
    df["age"] = (trans_time - dob).dt.days / 365.25

    # Haversine distance (km) between customer and merchant
    lat1, lon1 = np.radians(df["lat"]), np.radians(df["long"])
    lat2, lon2 = np.radians(df["merch_lat"]), np.radians(df["merch_long"])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    df["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

    return df[NUMERIC + CATEGORICAL]


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    columns = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("columns", columns),
    ])
