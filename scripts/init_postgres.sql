-- Multi-database initialization script for PostgreSQL
CREATE DATABASE airflow_db;
CREATE DATABASE dsp_app;
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO postgres;
GRANT ALL PRIVILEGES ON DATABASE dsp_app TO postgres;
