import os
from datetime import timedelta, datetime
import hydra
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from hydra import compose, initialize_config_dir


from sample_data import sample_data
from validate_data import validate_initial_data

BASE_PATH = os.environ["PYTHONPATH"]
def take_sample():
    os.chdir(BASE_PATH)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=f"{BASE_PATH}/../configs")
    cfg = compose(config_name="main")
    sample_data(cfg)

def validate():
    path = os.environ["PYTHONPATH"]
    os.chdir(path)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=f"{path}/../configs")
    cfg = compose(config_name="main")
    validate_initial_data(cfg)



dag = DAG(
    'data_extract_dag',
    schedule_interval=timedelta(minutes=5),
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

trigger = TriggerDagRunOperator(
    task_id='trigger_dagrun',
    trigger_dag_id='data_prepare_dag',
    dag=dag,
)

extract_task >> validate_task >> version_task >> load_task >> trigger
