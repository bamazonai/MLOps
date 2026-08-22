"""Preprocesamiento de inferencia: mismas 19 columnas del ETL y StandardScaler."""
import numpy as np

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


def apply_scaler(
    row: dict,
    mean: np.ndarray,
    std: np.ndarray,
    columns: list[str] | None = None,
) -> np.ndarray:
    """Escala una fila con (x - mean) / std, en el orden de las columnas del ETL."""
    cols = columns if columns is not None else FEATURE_COLUMNS
    values = np.array([row[col] for col in cols], dtype=float)
    return ((values - mean) / std).reshape(1, -1)


def log_to_dollars(y_log: float) -> float:
    """Invierte driver_pay_log: el modelo predice log, el API devuelve dólares."""
    return float(np.exp(y_log))
