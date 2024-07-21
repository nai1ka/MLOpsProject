from airflow import DAG

from datetime import timedelta, datetime
import pandas as pd
import great_expectations as ge
import os
from airflow.sensors.external_task import ExternalTaskSensor
import subprocess
import yaml
from hydra import compose, initialize,initialize_config_dir
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from sample_data import sample_data
from validate_data import validate_initial_data



dag = DAG(
    'data_prepare_dag',
    schedule=timedelta(minutes=5),
    start_date=datetime(2021, 1, 1),
    catchup=False,
)

extract_data_sensor = ExternalTaskSensor(
    dag=dag,
    task_id='extract_data_sensor',
    external_dag_id='data_extract_dag',
    timeout=300,
    poke_interval=20,
    mode='poke'
)

prepare_task = BashOperator(
    task_id='prepare_data_task',
    bash_command='cd $PROJECTPATH; python3 pipelines/data_prepare.py ',
    dag=dag,
)

extract_data_sensor >> prepare_task
