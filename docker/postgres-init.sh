#!/bin/bash
set -e
psql -v ON_ERROR_STOP=1 --username "admin" --dbname "postgres" <<-EOSQL
    CREATE DATABASE airflow;
    CREATE DATABASE mlflow;
    CREATE DATABASE datacenter_gold;
EOSQL
echo "Databases created."