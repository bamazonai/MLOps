"""Tests de feature engineering HVFHS (mismo criterio que el ETL)."""
import numpy as np
import pandas as pd

from hvfhs.features import FEATURE_COLUMNS, TARGET, build_features
from hvfhs.train import should_promote


def _raw_row(**overrides):
    row = {
        "hvfhs_license_num": "HV0003",
        "pickup_datetime": "2024-12-01 18:30:00",
        "trip_miles": 2.0,
        "trip_time": 600.0,
        "base_passenger_fare": 15.0,
        "driver_pay": 12.0,
        "tolls": np.nan,
        "bcf": 0.4,
        "sales_tax": 1.3,
        "congestion_surcharge": 0.0,
        "airport_fee": 0.0,
        "tips": 0.0,
        "shared_request_flag": "N",
        "shared_match_flag": "N",
        "access_a_ride_flag": "N",
        "wav_request_flag": "N",
        "wav_match_flag": "N",
    }
    row.update(overrides)
    return row


def test_build_features_adds_logs_time_and_flags():
    out = build_features(pd.DataFrame([_raw_row()]))

    assert list(out.columns) == FEATURE_COLUMNS + [TARGET]
    assert out["es_uber"].iloc[0] == 1
    assert out["shared_request_flag"].iloc[0] == 0
    assert out["hora"].iloc[0] == 18
    assert out["dia_semana"].iloc[0] == 6
    assert out["es_fin_de_semana"].iloc[0] == 1
    assert out["franja_horaria"].iloc[0] == 3
    assert out["tolls"].iloc[0] == 0
    np.testing.assert_allclose(out["trip_miles_log"].iloc[0], np.log(2.0))
    np.testing.assert_allclose(out["driver_pay_log"].iloc[0], np.log(12.0))


def test_build_features_drops_non_positive_trips():
    df = pd.DataFrame([_raw_row(), _raw_row(trip_miles=0.0)])
    out = build_features(df)
    assert len(out) == 1


def test_should_promote_when_there_is_no_champion():
    assert should_promote(challenger_r2=0.9, champion_r2=None) is True


def test_should_promote_if_challenger_is_at_least_as_good():
    assert should_promote(challenger_r2=0.98, champion_r2=0.98) is True
    assert should_promote(challenger_r2=0.97, champion_r2=0.98) is False
