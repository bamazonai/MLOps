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
- **Algoritmo:** **XGBoost**, el de mejor desempeño frente a Regresión Lineal, Ridge y
  Regresión Polinómica (R² ≈ 0.95 en el conjunto de prueba).

## 🏗️ Stack tecnológico

| Componente        | Rol                                                      |
|-------------------|----------------------------------------------------------|
| Apache Airflow    | Orquestación del pipeline (ETL y entrenamiento)          |
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
- **Primer contacto con MLflow** establecido desde el pipeline de ETL.

### 🚧 Pendiente / plan de continuación

1. **Experimento de MLflow con búsqueda de hiperparámetros** para XGBoost, logueando
   parámetros, métricas (R², RMSE, MAE) y registrando el mejor modelo en el Model
   Registry.
2. **DAG de entrenamiento** que consuma el parquet procesado y registre el modelo en
   MLflow.
3. **Servir el modelo mediante FastAPI**, con un endpoint de predicción que aplique el
   mismo preprocesamiento que en el entrenamiento (incluye deshacer la transformación
   logarítmica con `np.exp`).
4. **Modularizar** la lógica de las tareas del DAG.
5. **Documentación final** (README, docstrings y documentación automática de la API).

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