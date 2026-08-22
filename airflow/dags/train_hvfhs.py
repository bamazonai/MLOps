import datetime

from airflow.decorators import dag, task

markdown_text = """
### Entrenamiento HVFHS (champion / challenger)

Lee train/test ya escalados de MinIO, entrena **XGBoost** con la receta ganadora de la
notebook (`n_estimators=200`, `max_depth=6`, `learning_rate=0.1`) y lo registra en
MLflow como `hvfhs-driver-pay`.

Si el R² de test es mayor o igual al del alias **champion**, lo promueve. Si todavía
no hay champion (solo está la v1 de la notebook), este run pasa a serlo.

Hay que haber corrido antes el DAG `process_etl_hvfhs_data`.
"""

default_args = {
    "owner": "Grupo 8",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=45),
}


@dag(
    dag_id="train_hvfhs_model",
    description="Entrena XGBoost (HVFHS), lo registra y promociona champion si gana en R².",
    doc_md=markdown_text,
    schedule=None,
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
    tags=["Train", "HVFHS"],
    default_args=default_args,
)
def train_hvfhs_model():
    @task.virtualenv(
        task_id="train_and_register",
        requirements=[
            "awswrangler==3.6.0",
            "scikit-learn==1.3.2",
            "mlflow==2.10.2",
            "pandas==2.1.4",
            "pyarrow==15.0.0",
            "xgboost==2.0.3",
            "numpy==1.26.4",
        ],
        system_site_packages=True,
    )
    def train_and_register():
        """Entrena el challenger, lo registra y lo compara con el champion."""
        import sys

        sys.path.insert(0, "/opt/airflow/dags")
        import mlflow
        import numpy as np
        import awswrangler as wr
        from mlflow.models import infer_signature
        from hvfhs.constants import (
            BEST_XGB_PARAMS,
            EXPERIMENT_NAME,
            MLFLOW_TRACKING_URI,
            MODEL_NAME,
            X_TEST_PATH,
            X_TRAIN_PATH,
            Y_TEST_PATH,
            Y_TRAIN_PATH,
        )
        from hvfhs.train import (
            regression_metrics,
            should_promote,
            squeeze_target,
            train_xgb,
        )

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        X_train = wr.s3.read_parquet(path=X_TRAIN_PATH)
        X_test = wr.s3.read_parquet(path=X_TEST_PATH)
        y_train = squeeze_target(wr.s3.read_parquet(path=Y_TRAIN_PATH))
        y_test = squeeze_target(wr.s3.read_parquet(path=Y_TEST_PATH))

        model = train_xgb(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        usd = regression_metrics(np.exp(y_test), np.exp(y_pred))

        client = mlflow.MlflowClient()
        champion_r2 = None
        try:
            champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
            champ_model = mlflow.pyfunc.load_model(champ.source)
            champ_pred = np.asarray(champ_model.predict(X_test)).ravel()
            champion_r2 = regression_metrics(y_test, champ_pred)["r2"]
        except Exception as exc:
            print(f"No hay champion aún ({exc}). Se usará este run como primer champion si aplica.")

        experiment = mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name="xgb_challenger_airflow",
        ) as run:
            mlflow.log_params(BEST_XGB_PARAMS)
            mlflow.log_metrics({
                "r2": metrics["r2"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "usd_r2": usd["r2"],
                "usd_rmse": usd["rmse"],
                "usd_mae": usd["mae"],
            })
            if champion_r2 is not None:
                mlflow.log_metric("champion_r2", champion_r2)

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=MODEL_NAME,
            )
            run_id = run.info.run_id

        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        matching = [int(v.version) for v in versions if v.run_id == run_id]
        new_version = str(max(matching) if matching else max(int(v.version) for v in versions))

        client.set_registered_model_alias(MODEL_NAME, "challenger", new_version)

        promote = should_promote(metrics["r2"], champion_r2)
        if promote:
            client.set_registered_model_alias(MODEL_NAME, "champion", str(new_version))
            winner = "challenger"
        else:
            winner = "champion"

        print(
            f"Challenger R²={metrics['r2']:.4f} | Champion R²={champion_r2} | "
            f"winner={winner} | registered v{new_version}"
        )

    train_and_register()


dag = train_hvfhs_model()
