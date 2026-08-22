"""Split train/test y StandardScaler del ETL."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hvfhs.constants import RANDOM_STATE, TEST_SIZE
from hvfhs.features import TARGET


def split_xy(df: pd.DataFrame):
    """Partición 80/20 sin stratify (regresión)."""
    X = df.drop(columns=TARGET)
    y = df[[TARGET]]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def fit_transform_scaler(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Ajusta StandardScaler solo con train y transforma train y test."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    data_info = {
        "columns": X_train.columns.to_list(),
        "standard_scaler_mean": scaler.mean_.tolist(),
        "standard_scaler_std": scaler.scale_.tolist(),
    }
    return X_train_scaled, X_test_scaled, data_info
