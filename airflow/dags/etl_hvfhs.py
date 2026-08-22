import datetime

from airflow.decorators import dag, task

markdown_text = """
### ETL HVFHS

Descarga una muestra de viajes HVFHS (NYC TLC, diciembre 2024), limpia, arma features,
parte train/test y estandariza. Escribe los parquet en MinIO (`s3://data/`) y registra
el run `etl_normalize` en el experimento **HVFHS Driver Pay**.

La lógica vive en el paquete `hvfhs/` (este archivo solo orquesta).
"""

default_args = {
    "owner": "Grupo 8",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=60),
}


@dag(
    dag_id="process_etl_hvfhs_data",
    description="ETL de viajes HVFHS: descarga, features, split y scaler.",
    doc_md=markdown_text,
    schedule=None,
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
    tags=["ETL", "HVFHS"],
    default_args=default_args,
)
def process_etl_hvfhs_data():
    @task.virtualenv(
        task_id="get_data",
        requirements=["awswrangler==3.6.0", "pyarrow==15.0.0", "pandas==2.1.4"],
        system_site_packages=True,
    )
    def get_data():
        """Descarga el dataset de HVFHS, toma una muestra y guarda en S3."""
        import sys

        sys.path.insert(0, "/opt/airflow/dags")
        import awswrangler as wr
        from hvfhs.constants import RAW_PATH
        from hvfhs.ingest import download_sample

        df = download_sample()
        wr.s3.to_parquet(df=df, path=RAW_PATH)
        print(f"Muestramos lo guardado: {df.shape[0]:,} filas")

    @task.virtualenv(
        task_id="clean_and_features",
        requirements=["awswrangler==3.6.0", "pyarrow==15.0.0", "pandas==2.1.4", "numpy==1.26.4"],
        system_site_packages=True,
    )
    def clean_and_features():
        """Limpieza y feature engineering del dataset de HVFHS."""
        import sys

        sys.path.insert(0, "/opt/airflow/dags")
        import awswrangler as wr
        from hvfhs.constants import PROCESSED_PATH, RAW_PATH
        from hvfhs.features import build_features

        df = wr.s3.read_parquet(path=RAW_PATH)
        df = build_features(df)
        wr.s3.to_parquet(df=df, path=PROCESSED_PATH)
        print(f"Dataset limpio: {df.shape[0]:,} filas, {df.shape[1]} columnas")

    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0", "scikit-learn==1.3.2", "pandas==2.1.4", "pyarrow==15.0.0"],
        system_site_packages=True,
    )
    def split_dataset():
        """Separa el dataset limpio en train (80%) y test (20%)."""
        import sys

        sys.path.insert(0, "/opt/airflow/dags")
        import awswrangler as wr
        from hvfhs.constants import (
            PROCESSED_PATH,
            X_TEST_PATH,
            X_TRAIN_PATH,
            Y_TEST_PATH,
            Y_TRAIN_PATH,
        )
        from hvfhs.split_scale import split_xy

        df = wr.s3.read_parquet(path=PROCESSED_PATH)
        X_train, X_test, y_train, y_test = split_xy(df)
        wr.s3.to_parquet(df=X_train, path=X_TRAIN_PATH)
        wr.s3.to_parquet(df=X_test, path=X_TEST_PATH)
        wr.s3.to_parquet(df=y_train, path=Y_TRAIN_PATH)
        wr.s3.to_parquet(df=y_test, path=Y_TEST_PATH)
        print(f"Train: {X_train.shape[0]:,} filas | Test: {X_test.shape[0]:,} filas")

    @task.virtualenv(
        task_id="normalize_data",
        requirements=["awswrangler==3.6.0", "scikit-learn==1.3.2", "mlflow==2.10.2", "pandas==2.1.4", "pyarrow==15.0.0"],
        system_site_packages=True,
    )
    def normalize_data():
        """Estandariza las features y guarda los parámetros del scaler."""
        import json
        import sys

        sys.path.insert(0, "/opt/airflow/dags")
        import boto3
        import mlflow
        import awswrangler as wr
        from hvfhs.constants import (
            DATA_INFO_BUCKET,
            DATA_INFO_KEY,
            EXPERIMENT_NAME,
            MLFLOW_TRACKING_URI,
            X_TEST_PATH,
            X_TRAIN_PATH,
        )
        from hvfhs.split_scale import fit_transform_scaler

        X_train = wr.s3.read_parquet(path=X_TRAIN_PATH)
        X_test = wr.s3.read_parquet(path=X_TEST_PATH)
        X_train, X_test, data_dict = fit_transform_scaler(X_train, X_test)
        wr.s3.to_parquet(df=X_train, path=X_TRAIN_PATH)
        wr.s3.to_parquet(df=X_test, path=X_TEST_PATH)

        boto3.client("s3").put_object(
            Bucket=DATA_INFO_BUCKET,
            Key=DATA_INFO_KEY,
            Body=json.dumps(data_dict, indent=2),
        )

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment = mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name="etl_normalize", experiment_id=experiment.experiment_id):
            mlflow.log_param("Train observations", X_train.shape[0])
            mlflow.log_param("Test observations", X_test.shape[0])

        print("Scaler guardado en data_info/data.json y registrado en MLflow")

    get_data() >> clean_and_features() >> split_dataset() >> normalize_data()


dag = process_etl_hvfhs_data()
