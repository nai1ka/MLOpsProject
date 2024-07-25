from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator


dag = DAG(
    'data_prepare_dag',
    schedule_interval=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
)

prepare_task = BashOperator(
    task_id='prepare_data_task',
    bash_command='cd $PROJECTPATH; python3 pipelines/data_prepare.py ',
    dag=dag,
)

prepare_task
