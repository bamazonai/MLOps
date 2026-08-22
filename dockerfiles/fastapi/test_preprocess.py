"""Tests del preprocesamiento de inferencia (scaler + vuelta a dólares)."""
import numpy as np
import pytest

from preprocess import FEATURE_COLUMNS, apply_scaler, log_to_dollars


def test_feature_columns_are_the_19_etl_features():
    assert FEATURE_COLUMNS == [
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


def test_apply_scaler_is_standard_score():
    row = {col: 10.0 for col in FEATURE_COLUMNS}
    mean = np.zeros(len(FEATURE_COLUMNS))
    std = np.ones(len(FEATURE_COLUMNS)) * 2.0

    scaled = apply_scaler(row, mean=mean, std=std)

    assert scaled.shape == (1, 19)
    np.testing.assert_allclose(scaled, 5.0)


def test_apply_scaler_uses_column_order():
    row = {col: 0.0 for col in FEATURE_COLUMNS}
    row["hora"] = 8.0
    mean = np.zeros(len(FEATURE_COLUMNS))
    std = np.ones(len(FEATURE_COLUMNS))
    hora_idx = FEATURE_COLUMNS.index("hora")

    scaled = apply_scaler(row, mean=mean, std=std)

    assert scaled[0, hora_idx] == pytest.approx(8.0)
    assert scaled[0, 0] == pytest.approx(0.0)


def test_log_to_dollars_inverts_np_log():
    assert log_to_dollars(np.log(25.0)) == pytest.approx(25.0)
