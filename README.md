# 🚕 Predicción del pago al conductor en viajes HVFHS de NYC

Trabajo Práctico Final — **Operaciones de Aprendizaje de Máquina I (MLOps1)**
Especialización en Inteligencia Artificial (CEIA) — FIUBA

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

## 📊 Estado de avance (entrega parcial)

Completado a la fecha:

-**Infraestructura productiva desplegada y funcional.** Los 10 contenedores
  levantan correctamente y en estado *healthy*. Se verificó el acceso a las cuatro
  interfaces: Airflow, MLflow, MinIO y FastAPI.
-**Repositorio del grupo creado y versionado en GitHub.**
-**DAG de ETL implementado y en ejecución.** Dos de sus cuatro tareas ya corren
  correctamente de punta a punta en Airflow, escribiendo sus resultados en MinIO:
  1. `get_data`— descarga los datos de la NYC TLC, toma una muestra de 500k filas
     (lectura por bloques para no saturar memoria) y la guarda en `s3://data/raw/`.
  2. `clean_and_features`— limpieza, transformación logarítmica y feature
     engineering; guarda el dataset procesado en `s3://data/processed/`.
  3. `split_dataset` — partición train/test (pendiente).
  4. `normalize_data` — estandarización y persistencia del scaler (pendiente).

## 🚧 Trabajo pendiente y plan de continuación

1. **Completar las dos tareas restantes del ETL** (`split_dataset` y `normalize_data`).
2. **Implementar el experimento de MLflow con búsqueda de hiperparámetros** para
   XGBoost, registrando corridas, métricas y el mejor modelo en el Model Registry.
3. **Servir el modelo mediante FastAPI**, con un endpoint de predicción que aplique el
   mismo preprocesamiento que en el entrenamiento.
4. **Completar la documentación** (README final, docstrings y documentación automática
   de la API).

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