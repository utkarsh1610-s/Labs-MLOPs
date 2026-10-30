FROM apache/airflow:3.3.2

# Pinning apache-airflow in the same install stops pip from upgrading or
# downgrading Airflow itself while it resolves the extra packages.
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
