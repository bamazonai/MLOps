"""Constantes compartidas por el ETL y el DAG de entrenamiento HVFHS."""

EXPERIMENT_NAME = "HVFHS Driver Pay"
MODEL_NAME = "hvfhs-driver-pay"
MLFLOW_TRACKING_URI = "http://mlflow:5000"

RAW_PATH = "s3://data/raw/hvfhs_raw.parquet"
PROCESSED_PATH = "s3://data/processed/hvfhs_cleaned.parquet"
X_TRAIN_PATH = "s3://data/final/train/hvfhs_X_train.parquet"
X_TEST_PATH = "s3://data/final/test/hvfhs_X_test.parquet"
Y_TRAIN_PATH = "s3://data/final/train/hvfhs_y_train.parquet"
Y_TEST_PATH = "s3://data/final/test/hvfhs_y_test.parquet"
DATA_INFO_BUCKET = "data"
DATA_INFO_KEY = "data_info/data.json"

TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2024-12.parquet"
SAMPLE_SIZE = 500_000
RANDOM_STATE = 42
TEST_SIZE = 0.2

RAW_COLUMNS = [
    "hvfhs_license_num",
    "pickup_datetime",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "driver_pay",
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
]

BEST_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}
