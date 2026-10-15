"""Airflow DAG kept import-safe when Airflow is not installed locally."""

from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator

    with DAG(
        dag_id="video_recsys_daily",
        start_date=datetime(2026, 1, 1),
        schedule="0 3 * * *",
        catchup=False,
        max_active_runs=1,
        tags=["recsys", "video"],
    ) as dag:
        features = BashOperator(
            task_id="build_features",
            bash_command="spark-submit jobs/build_features.py --date {{ ds }}",
        )
        train = BashOperator(
            task_id="train_ranker",
            bash_command="video-recsys --train-steps 200 --artifact artifacts/{{ ds }}/model.pt",
        )
        validate = BashOperator(
            task_id="validate_model",
            bash_command="python -m pytest tests/test_model.py tests/test_metrics.py",
        )
        features >> train >> validate
except ImportError:
    dag = None

