import json
import os

import boto3
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from preprocess import apply_scaler, log_to_dollars

MODEL_NAME = os.getenv("MODEL_NAME", "hvfhs-driver-pay")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
S3_ENDPOINT = os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000")


def load_model(model_name: str, version: str):
    """Carga el champion del Registry; si no hay alias, usa la versión numérica."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, MODEL_ALIAS)
        model_ml = mlflow.pyfunc.load_model(mv.source)
        version_model_ml = int(mv.version)
    except Exception:
        model_ml = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
        version_model_ml = int(version)

    s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT)
    raw = s3.get_object(Bucket="data", Key="data_info/data.json")["Body"].read().decode()
    data_dictionary = json.loads(raw)
    data_dictionary["standard_scaler_mean"] = np.array(
        data_dictionary["standard_scaler_mean"], dtype=float
    )
    data_dictionary["standard_scaler_std"] = np.array(
        data_dictionary["standard_scaler_std"], dtype=float
    )
    return model_ml, version_model_ml, data_dictionary


class ModelInput(BaseModel):
    """Features ya ingenierizadas (sin StandardScaler). Mismo esquema que el ETL."""

    trip_miles_log: float = Field(description="log(trip_miles)")
    trip_time_log: float = Field(description="log(trip_time) en segundos")
    base_passenger_fare_log: float = Field(description="log(base_passenger_fare)")
    tolls: float = Field(ge=0)
    bcf: float = Field(ge=0)
    sales_tax: float = Field(ge=0)
    congestion_surcharge: float = Field(ge=0)
    airport_fee: float = Field(ge=0)
    tips: float = Field(ge=0)
    hora: int = Field(ge=0, le=23, description="Hora de pickup (0-23)")
    dia_semana: int = Field(ge=0, le=6, description="Lunes=0 ... Domingo=6")
    es_fin_de_semana: int = Field(ge=0, le=1)
    franja_horaria: int = Field(ge=0, le=3, description="0=noche, 1=mañana, 2=tarde, 3=noche")
    es_uber: int = Field(ge=0, le=1, description="1 si hvfhs_license_num == HV0003")
    shared_request_flag: int = Field(ge=0, le=1)
    shared_match_flag: int = Field(ge=0, le=1)
    access_a_ride_flag: int = Field(ge=0, le=1)
    wav_request_flag: int = Field(ge=0, le=1)
    wav_match_flag: int = Field(ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "trip_miles_log": 1.2641,
                    "trip_time_log": 6.6464,
                    "base_passenger_fare_log": 2.722,
                    "tolls": 0,
                    "bcf": 0.42,
                    "sales_tax": 1.35,
                    "congestion_surcharge": 0,
                    "airport_fee": 0,
                    "tips": 0,
                    "hora": 18,
                    "dia_semana": 6,
                    "es_fin_de_semana": 1,
                    "franja_horaria": 3,
                    "es_uber": 1,
                    "shared_request_flag": 0,
                    "shared_match_flag": 0,
                    "access_a_ride_flag": 0,
                    "wav_request_flag": 0,
                    "wav_match_flag": 0,
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    driver_pay: float = Field(description="Pago al conductor en dólares (np.exp de la predicción)")
    driver_pay_log: float = Field(description="Predicción cruda del modelo, en log")
    model_version: int = Field(description="Versión del modelo en el Model Registry")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "driver_pay": 12.29,
                    "driver_pay_log": 2.5088,
                    "model_version": 1,
                }
            ]
        }
    }


model, version_model, data_dict = load_model(MODEL_NAME, MODEL_VERSION)

app = FastAPI(
    title="HVFHS Driver Pay",
    description="Predicción del pago al conductor en viajes HVFHS de NYC.",
)


@app.get("/")
def read_root():
    return JSONResponse(
        content=jsonable_encoder(
            {
                "message": "HVFHS Driver Pay API",
                "model": MODEL_NAME,
                "version": version_model,
                "alias": MODEL_ALIAS,
            }
        )
    )


@app.post("/predict", response_model=ModelOutput)
def predict(features: ModelInput):
    """Escala las features con el scaler del ETL, predice driver_pay_log y devuelve dólares."""
    scaled = apply_scaler(
        features.model_dump(),
        mean=data_dict["standard_scaler_mean"],
        std=data_dict["standard_scaler_std"],
        columns=data_dict["columns"],
    )
    X = pd.DataFrame(scaled, columns=data_dict["columns"])
    y_log = float(np.asarray(model.predict(X)).reshape(-1)[0])
    return ModelOutput(
        driver_pay=round(log_to_dollars(y_log), 2),
        driver_pay_log=round(y_log, 4),
        model_version=version_model,
    )
