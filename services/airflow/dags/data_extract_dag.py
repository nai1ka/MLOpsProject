from airflow import DAG

from datetime import timedelta, datetime
import pandas as pd
import great_expectations as ge
import os
import subprocess
import yaml
from hydra import compose, initialize,initialize_config_dir
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from sample_data import sample_data
from validate_data import validate_initial_data
def take_sample():
    # TODO maybe find a better way to navigate
    path = os.environ["PYTHONPATH"]
    os.chdir(path)
    initialize_config_dir(config_dir=f"{path}/../configs")
    cfg = compose(config_name="main")
    # Run sample_data function with Hydra configuration
    sample_data(cfg)

def validate():
    path = os.environ["PYTHONPATH"]
    os.chdir(path)
    initialize_config_dir(config_dir=f"{path}/../configs")
    cfg = compose(config_name="main")
    validate_initial_data(cfg)



dag = DAG(
    'data_extract_dag',
    schedule=timedelta(minutes=5),
    start_date=datetime(2021, 1, 1),
    catchup=False,
)

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=take_sample,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate,
    dag=dag,
)
version_task = BashOperator(
    task_id='version_data',
    bash_command='cd $AIRFLOW_HOME; bash ../../scripts/version_data.sh ',
    dag=dag,
)

load_task = BashOperator(
    task_id='load_data',
    bash_command='cd $PYTHONPATH; dvc push ',
    dag=dag,
)

extract_task >> validate_task >> version_task >> load_task
