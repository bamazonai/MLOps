import datetime
from airflow.decorators import dag, task

@dag(
    dag_id = "process_etl_hvfhs_data",
    schedule=None,
    start_date = datetime.datetime(2024, 1, 1),
    catchup=False,
    tags = ["ETL", "HVFHS"],
)
def process_etl_hvfhs_data():
    @task.virtualenv(
        task_id = "get_data",
        requirements=["awswrangler==3.6.0", "pyarrow==15.0.0", "pandas==2.1.4"],
        system_site_packages=True,
    )
    def get_data():
        """ Descarga el dataset de HVFHS, selecciona columnas y guarda en S3"""
        import urllib.request
        import pyarrow.parquet as pq
        import pyarrow as pa
        import pandas as pd
        import awswrangler as wr

        url = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2024-12.parquet"

        columnas = [
            "hvfhs_license_num", "pickup_datetime",
            "trip_miles", "trip_time", "base_passenger_fare",
            "tolls", "bcf", "sales_tax", "congestion_surcharge", "airport_fee", "tips",
            "driver_pay",
            "shared_request_flag", "shared_match_flag", "access_a_ride_flag",
            "wav_request_flag", "wav_match_flag",
        ]

        #Descargamos pero para que no se muera, usemos disco y no memoria
        local_path = "/tmp/fhvhv.parquet"
        urllib.request.urlretrieve(url, local_path)

        #Leeremos solo los primeros 500,000 registros para no saturar la memoria
        pf = pq.ParquetFile(local_path)
        frames, total = [], 0

        for rg in range(pf.num_row_groups):
            frames.append(pf.read_row_group(rg, columns=columnas))
            total += frames[-1].num_rows
            if total >= 500_000:
                break

        df = pa.concat_tables(frames).to_pandas()

        #Muestra 500K
        df = df.sample(n=min(500_000, df.shape[0]), random_state=42).reset_index(drop=True)
        wr.s3.to_parquet(df=df, path="s3://data/raw/hvfhs_raw.parquet")

        print(f"Muestramos lo guardado: {df.shape[0]:,} filas")

    @task.virtualenv(task_id = "clean_and_features", requirements=["awswrangler==3.6.0", "pyarrow==15.0.0", "pandas==2.1.4", "numpy==1.26.4"], system_site_packages=True)
    def clean_and_features():
        """Limpieza y feature engineering del dataset de HVFHS"""
        import numpy as np
        import pandas as pd
        import awswrangler as wr

        #Ahora si, ya cargados los datos, limpiamos y hacemos feature engineering
        df = wr.s3.read_parquet(path="s3://data/raw/hvfhs_raw.parquet")

        #Limpieza, solo viajes con valores positivos
        for col in ["trip_miles", "trip_time", "base_passenger_fare", "driver_pay"]:
            df = df[df[col] > 0]

        # Fees que hayan sido nulos
        for col in ["tolls", "bcf", "sales_tax", "congestion_surcharge", "airport_fee", "tips"]:
            df[col] = df[col].fillna(0)

        #Aplicamos la trasformacion logaritmica a las variables continuas
        df["trip_miles_log"] = np.log(df["trip_miles"])
        df["trip_time_log"] = np.log(df["trip_time"])
        df["base_passenger_fare_log"] = np.log(df["base_passenger_fare"])
        df["driver_pay_log"] = np.log(df["driver_pay"])

        #Features de tiempo
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
        df["hora"] = df["pickup_datetime"].dt.hour
        df["dia_semana"] = df["pickup_datetime"].dt.dayofweek
        df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
        df["franja_horaria"] = pd.cut(df["hora"], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], right=False).astype(int)


        #Ahora nos fijamos en el encoding
        df["es_uber"] = (df["hvfhs_license_num"] == "HV0003").astype(int)
        flags = ["shared_request_flag", "shared_match_flag", "access_a_ride_flag", "wav_request_flag", "wav_match_flag"]
        for flag in flags:
            df[flag] = (df[flag] == "Y").astype(int)

        #Quedammos con los features y target
        features = ["trip_miles_log", "trip_time_log", "base_passenger_fare_log",
                    "tolls", "bcf", "sales_tax", "congestion_surcharge", "airport_fee", "tips",
                    "hora", "dia_semana", "es_fin_de_semana", "franja_horaria",
                    "es_uber", "shared_request_flag", "shared_match_flag",
                    "access_a_ride_flag", "wav_request_flag", "wav_match_flag"]
        
        target = "driver_pay_log"

        df = df[features + [target]].dropna().reset_index(drop=True)
    
        #Guardamos el dataset limpio y con features
        wr.s3.to_parquet(df=df, path="s3://data/processed/hvfhs_cleaned.parquet")
        print(f"Dataset limpio: {df.shape[0]:,} filas, {df.shape[1]} columnas")

    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0", "scikit-learn==1.3.2", "pandas==2.1.4", "pyarrow==15.0.0"],
        system_site_packages=True,
    )
    def split_dataset():
        """Separa el dataset limpio en train (80%) y test (20%)."""
        import awswrangler as wr
        from sklearn.model_selection import train_test_split

        # Leer el dataset limpio que dejó clean_and_features
        df = wr.s3.read_parquet(path="s3://data/processed/hvfhs_cleaned.parquet")

        target = "driver_pay_log"
        X = df.drop(columns=target)   # las 19 features
        y = df[[target]]              # el target (lo dejamos como DataFrame)

        # Regresión -> sin stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Guardar las 4 partes en S3
        wr.s3.to_parquet(df=X_train, path="s3://data/final/train/hvfhs_X_train.parquet")
        wr.s3.to_parquet(df=X_test,  path="s3://data/final/test/hvfhs_X_test.parquet")
        wr.s3.to_parquet(df=y_train, path="s3://data/final/train/hvfhs_y_train.parquet")
        wr.s3.to_parquet(df=y_test,  path="s3://data/final/test/hvfhs_y_test.parquet")

        print(f"Train: {X_train.shape[0]:,} filas | Test: {X_test.shape[0]:,} filas")
    @task.virtualenv(
        task_id="normalize_data",
        requirements=["awswrangler==3.6.0", "scikit-learn==1.3.2",
                      "mlflow==2.10.2", "pandas==2.1.4", "pyarrow==15.0.0"],
        system_site_packages=True,
    )
    def normalize_data():
        """Estandariza las features y guarda los parámetros del scaler."""
        import json
        import boto3
        import mlflow
        import pandas as pd
        import awswrangler as wr
        from sklearn.preprocessing import StandardScaler

        # Leer train y test que dejó split_dataset
        X_train = wr.s3.read_parquet(path="s3://data/final/train/hvfhs_X_train.parquet")
        X_test = wr.s3.read_parquet(path="s3://data/final/test/hvfhs_X_test.parquet")

        # Ajustar el scaler SOLO con train, transformar ambos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Volver a DataFrame (fit_transform devuelve un array sin nombres de columna)
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        # Sobrescribir en S3 las versiones ya escaladas
        wr.s3.to_parquet(df=X_train, path="s3://data/final/train/hvfhs_X_train.parquet")
        wr.s3.to_parquet(df=X_test,  path="s3://data/final/test/hvfhs_X_test.parquet")

        # Guardar los parámetros del scaler (media y desvío) en un JSON en S3
        data_dict = {
            "columns": X_train.columns.to_list(),
            "standard_scaler_mean": scaler.mean_.tolist(),
            "standard_scaler_std": scaler.scale_.tolist(),
        }
        client = boto3.client("s3")
        client.put_object(
            Bucket="data",
            Key="data_info/data.json",
            Body=json.dumps(data_dict, indent=2),
        )

        # Primer registro en MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        experiment = mlflow.set_experiment("HVFHS Driver Pay")
        with mlflow.start_run(run_name="etl_normalize", experiment_id=experiment.experiment_id):
            mlflow.log_param("Train observations", X_train.shape[0])
            mlflow.log_param("Test observations", X_test.shape[0])

        print("Scaler guardado en data_info/data.json y registrado en MLflow")

    get_data() >> clean_and_features() >> split_dataset() >> normalize_data()

dag = process_etl_hvfhs_data()