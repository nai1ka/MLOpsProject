#!/bin/bash

echo $PWD

usage() {
    echo "Usage: $0 -m MODEL_NAME -a MODEL_ALIAS"
    exit 1
}

# Parse command-line arguments
while getopts "m:a:" opt; do
  case $opt in
    m) model_name="$OPTARG"
    ;;
    a) model_alias="$OPTARG"
    ;;
    *) usage
    ;;
  esac
done

mlflow models generate-dockerfile --model-uri models:/${model_name}@${model_alias} --env-manager local -d api


cd api

docker build -t predict_taxi_price_ml_service .
docker run --rm -p 5151:8080 predict_taxi_price_ml_service -d
docker tag my_ml_service nai1ka/predict_taxi_price_ml_service
docker push nai1ka/predict_taxi_price_ml_service:latest