-- The postgres image auto-creates the database named by POSTGRES_DB
-- (dsp_db, our application database). This script additionally creates
-- a separate database for Airflow's own metadata, as required:
-- "single PostgreSQL instance with separate databases: one for your
-- application data ... and one for Airflow metadata".
CREATE DATABASE airflow_db;
