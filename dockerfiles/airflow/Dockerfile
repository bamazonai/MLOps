FROM apache/airflow:3.0.2
COPY requirements.txt /requirements.txt

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         git \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir apache-airflow[virtualenv]==3.0.2 -r /requirements.txt

# /tmp/.cache gets created above owned by airflow (uid 50000) with mode 755.
# Containers run with an arbitrary AIRFLOW_UID (see AIRFLOW_GID=0 in .env), so
# uv (used by @task.virtualenv) fails with "Permission denied" creating
# /tmp/.cache/uv unless the group (0) also has write access.
USER root
RUN chmod -R g+rwX /tmp/.cache
USER airflow
