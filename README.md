# 🚕 Predicción del pago al conductor en viajes HVFHS de NYC

Trabajo Práctico Final — **Operaciones de Aprendizaje de Máquina I (MLOps1)**
Especialización en Inteligencia Artificial (CEIA) — FIUBA

**Grupo 8**
**Autores:** Byron Garcia · María Gabriela Di Grazia

---

## 🧭 Objetivo del proyecto

Integrar un modelo de Machine Learning —desarrollado previamente en la materia de
Aprendizaje de Máquina— dentro de una infraestructura productiva de MLOps.

El modelo predice el **pago al conductor (`driver_pay`)** de viajes de transporte de
alta demanda (*High Volume For-Hire Services*, HVFHS) de la ciudad de Nueva York, a
partir del dataset público de la NYC TLC (diciembre 2024).

La infraestructura se basa en el ambiente productivo provisto por la cátedra
(`amq2-service-ml`), orquestado con Docker Compose.

## 🧠 Modelo de base

- **Tipo de problema:** regresión.
- **Variable a predecir:** `driver_pay` (en escala logarítmica).
- **Preprocesamiento:** filtrado de outliers, transformación logarítmica de variables
  sesgadas, feature engineering temporal (hora, día de semana, franja horaria, fin de
  semana), codificación de variables categóricas y flags, y estandarización con
  `StandardScaler`.
- **Algoritmo:** **XGBoost**. Tras la búsqueda de hiperparámetros, el mejor modelo
  (`n_estimators=200`, `max_depth=6`, `learning_rate=0.1`) alcanzó R² ≈ 0.983 en test
  (escala logarítmica).

## 🏗️ Stack tecnológico

| Componente        | Rol                                                      |
|-------------------|----------------------------------------------------------|
| Apache Airflow    | Orquestación del pipeline ETL                            |
| MLflow            | Registro de experimentos y versionado de modelos         |
| FastAPI           | Servicio de inferencia (API REST)                        |
| MinIO             | Almacenamiento de datos y artefactos (compatible con S3) |
| PostgreSQL        | Base de datos de Airflow y MLflow                        |
| ValKey            | Backend de mensajería de Airflow                         |

---

## 📊 Estado de avance

### ✅ Completado

- **Infraestructura productiva desplegada y funcional.** Los 10 contenedores levantan
  correctamente y en estado *healthy*. Verificado el acceso a Airflow, MLflow, MinIO y
  FastAPI.
- **Repositorio del grupo creado y versionado en GitHub.**
- **DAG de ETL completo y en ejecución.** Las cuatro tareas corren de punta a punta en
  Airflow, escribiendo sus resultados en MinIO:
  1. `get_data` — descarga los datos de la NYC TLC, toma una muestra de 500k filas
     (lectura por bloques para no saturar memoria) y la guarda en `s3://data/raw/`.
  2. `clean_and_features` — limpieza, transformación logarítmica y feature engineering;
     guarda el dataset procesado en `s3://data/processed/`.
  3. `split_dataset` — partición train/test (80/20) en `s3://data/final/`.
  4. `normalize_data` — estandarización con `StandardScaler` (ajustado solo con train),
     persistencia de los parámetros del scaler en `s3://data/data_info/data.json` y
     primer registro en MLflow (experimento *HVFHS Driver Pay*).
- **Experimento de MLflow.** Notebook `notebook_example/experiment_mlflow.ipynb`:
  grilla de 8 combinaciones de XGBoost (runs anidados), métricas R² / RMSE / MAE
  y registro del mejor modelo en el Model Registry como `hvfhs-driver-pay` v1.
- **API de inferencia (FastAPI).** `POST /predict` carga el modelo del Registry,
  aplica el `StandardScaler` del ETL (`s3://data/data_info/data.json`) y devuelve
  el pago en dólares (`np.exp` sobre `driver_pay_log`). Documentación interactiva
  en http://localhost:8800/docs.

### Cómo probar la API

Con el entorno levantado, abrir http://localhost:8800/docs → **POST /predict** →
**Try it out**. El cuerpo son las 19 features **ya ingenierizadas y sin escalar**
(las mismas que deja `clean_and_features`, no las de `final/` que ya están
estandarizadas).

Ejemplo:

```json
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
  "wav_match_flag": 0
}
```

Respuesta: `driver_pay` (USD), `driver_pay_log` y `model_version`.

Tras cambiar el código de FastAPI hay que reconstruir la imagen:

```bash
docker compose --profile all up -d --build fastapi
```

### 🚧 Extra (no exigido por la cátedra)

- **DAG de entrenamiento** `train_hvfhs_model`: lee train/test de MinIO, fitea XGBoost
  con la receta ganadora, registra una nueva versión de `hvfhs-driver-pay` y promociona
  el alias **champion** si el R² de test es mayor o igual al del champion actual.
- **ETL modularizado.** El DAG `process_etl_hvfhs_data` solo orquesta; la lógica está
  en `airflow/dags/hvfhs/` (`features`, `ingest`, `split_scale`, `train`).

Para correr el entrenamiento (después del ETL): Airflow → `train_hvfhs_model` → unpause
→ Trigger. En MLflow, Model registry → `hvfhs-driver-pay` deberían verse los alias
`challenger` y `champion`.

---

## ⚙️ Cómo levantar el entorno

Requisitos: [Docker](https://docs.docker.com/get-docker/) instalado y en ejecución.

```bash
# 1. Clonar el repositorio
git clone <URL-de-este-repo>
cd <carpeta-del-repo>

# 2. Crear las carpetas necesarias para Airflow
mkdir -p airflow/config airflow/dags airflow/logs airflow/plugins

# 3. (Linux/Mac) Ajustar el AIRFLOW_UID en el archivo .env con el resultado de:
id -u

# 4. Levantar todos los servicios
docker compose --profile all up
```

Interfaces disponibles una vez levantado el entorno:

| Servicio | URL                        |
|----------|----------------------------|
| Airflow  | http://localhost:8080      |
| MLflow   | http://localhost:5001      |
| MinIO    | http://localhost:9001      |
| FastAPI  | http://localhost:8800/docs |

---

## 📁 Estructura del pipeline de datos (MinIO / S3)

```
s3://data/
├── raw/hvfhs_raw.parquet              # muestra cruda descargada de NYC TLC
├── processed/hvfhs_cleaned.parquet    # dataset limpio + features
├── final/
│   ├── train/  (X_train, y_train)     # partición de entrenamiento (escalada)
│   └── test/   (X_test, y_test)       # partición de prueba (escalada)
└── data_info/data.json               # parámetros del StandardScaler
```