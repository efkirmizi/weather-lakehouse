#!/usr/bin/env bash
#
# Developer entry point. Plain bash on purpose: the job images are launched by
# DockerOperator rather than Compose, so building and testing needs a few docker
# invocations that would otherwise live only in someone's shell history.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"

JOB_IMAGES=(
    "dag-pyspark-etl:1.0|jobs/weather_etl"
    "dag-pyspark-feature-engineering:1.0|jobs/feature_engineering"
    "dag-pytorch-model-training:1.0|jobs/model_training"
    "dag-lakehouse-janitor:1.0|jobs/maintenance"
    "dag-nessie-gc:1.0|jobs/nessie_gc"
)

# Each suite runs inside the image that already carries its dependencies, so pytest
# never has to be baked into a production image.
run_pytest() {  # image, pythonpath, extra-args..., then test paths
    local image="$1" pythonpath="$2"; shift 2
    docker run --rm --user root \
        -v "${ROOT}/tests":/tests \
        -e PYTHONPATH="${pythonpath}" \
        "${image}" \
        sh -c "pip install -q pytest && python -m pytest $* -q -p no:cacheprovider"
}

build_jobs() {
    for spec in "${JOB_IMAGES[@]}"; do
        local tag="${spec%%|*}" ctx="${spec##*|}"
        echo ">> building ${tag}"
        docker build -q -t "${tag}" "${ctx}" >/dev/null
    done
}

test_static() {
    echo ">> static checks (discovery, import shadowing, syntax)"
    docker run --rm -v "${ROOT}":/repo -w /repo python:3.11-slim \
        sh -c "pip install -q pytest && python -m pytest tests/static -q"
}

test_dags() {
    echo ">> DAG folder parsed the way the scheduler parses it"
    docker run --rm \
        -v "${ROOT}/dags":/opt/airflow/dags \
        -v "${ROOT}/tests":/tests \
        -e AIRFLOW__CORE__LOAD_EXAMPLES=False \
        lakehouse-airflow-scheduler:latest \
        bash -c "pip install -q pytest && python -m pytest /tests/airflow -q -p no:cacheprovider"
}

test_unit() {
    echo ">> unit tests: training image"
    run_pytest dag-pytorch-model-training:1.0 /app \
        /tests/unit/test_data_loader.py /tests/unit/test_trainer.py \
        /tests/unit/test_drift_monitor.py /tests/unit/test_baselines.py \
        /tests/unit/test_promotion.py
    echo ">> unit tests: weather ETL image"
    run_pytest dag-pyspark-etl:1.0 /opt/spark/work-dir \
        /tests/unit/test_weather_etl.py
    echo ">> unit tests: feature engineering image"
    run_pytest dag-pyspark-feature-engineering:1.0 /opt/spark/work-dir \
        /tests/unit/test_feature_engineering.py
    echo ">> unit tests: serving image"
    run_pytest lakehouse-lakehouse-serving:latest /app \
        /tests/unit/test_serving.py
}

case "${1:-help}" in
    build)        build_jobs; docker compose build ;;
    build-jobs)   build_jobs ;;
    up)           docker compose up -d ;;
    down)         docker compose down ;;
    test)         test_static; test_dags; test_unit ;;
    test-static)  test_static ;;
    test-dags)    test_dags ;;
    test-unit)    test_unit ;;
    *)
        cat <<'USAGE'
usage: ./dev.sh <command>

  build        build the job images and the Compose services
  build-jobs   build only the DockerOperator job images
  up | down    start / stop the stack
  test         static + DAG parsing + unit tests
  test-static  dependency-free checks (run these first, they are instant)
  test-dags    parse the DAG folder exactly as the scheduler does
  test-unit    unit tests, each inside the image holding its dependencies
USAGE
        ;;
esac
