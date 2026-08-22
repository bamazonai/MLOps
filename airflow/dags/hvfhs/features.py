"""Limpieza y feature engineering del dataset HVFHS."""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "trip_miles_log",
    "trip_time_log",
    "base_passenger_fare_log",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "hora",
    "dia_semana",
    "es_fin_de_semana",
    "franja_horaria",
    "es_uber",
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
]

TARGET = "driver_pay_log"
FLAG_COLUMNS = [
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra outliers, aplica log, arma features temporales/flags y deja 19 columnas + target."""
    out = df.copy()

    for col in ["trip_miles", "trip_time", "base_passenger_fare", "driver_pay"]:
        out = out[out[col] > 0]

    for col in ["tolls", "bcf", "sales_tax", "congestion_surcharge", "airport_fee", "tips"]:
        out[col] = out[col].fillna(0)

    out["trip_miles_log"] = np.log(out["trip_miles"])
    out["trip_time_log"] = np.log(out["trip_time"])
    out["base_passenger_fare_log"] = np.log(out["base_passenger_fare"])
    out["driver_pay_log"] = np.log(out["driver_pay"])

    out["pickup_datetime"] = pd.to_datetime(out["pickup_datetime"])
    out["hora"] = out["pickup_datetime"].dt.hour
    out["dia_semana"] = out["pickup_datetime"].dt.dayofweek
    out["es_fin_de_semana"] = out["dia_semana"].isin([5, 6]).astype(int)
    out["franja_horaria"] = pd.cut(
        out["hora"], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], right=False
    ).astype(int)

    out["es_uber"] = (out["hvfhs_license_num"] == "HV0003").astype(int)
    for flag in FLAG_COLUMNS:
        out[flag] = (out[flag] == "Y").astype(int)

    return out[FEATURE_COLUMNS + [TARGET]].dropna().reset_index(drop=True)
