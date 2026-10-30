#!/usr/bin/env bash
# One-time setup for Airflow Lab 3. Run from the Airflow_Lab3 folder:  ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

# Folders Docker mounts into the containers
mkdir -p ./logs ./plugins ./config ./working_data

# Run containers as your user so files they write (logs, models) aren't owned by root
# FERNET_KEY is left blank on purpose (fine for a local lab); setting it silences Compose warnings
if [[ ! -f .env ]]; then
  printf "AIRFLOW_UID=%s\nFERNET_KEY=\n" "$(id -u)" > .env
  echo "Created .env (AIRFLOW_UID=$(id -u))"
fi

# Build the image (Airflow 3.3.2 + kneed + pytest), then create the DB and admin user
docker compose build
docker compose up airflow-init

echo
echo "Setup done. Start Airflow with:  docker compose up -d"
