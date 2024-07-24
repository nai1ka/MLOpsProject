#!/bin/bash

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

cd $PROJECTPATH

# Generate Dockerfile in the api directory
mlflow models generate-dockerfile --model-uri models:/${model_name}@${model_alias} --env-manager local -d api

cd api

# Build docker image
sudo docker build -t predict_taxi_price_ml_service .
# Tag docker image
sudo docker tag predict_taxi_price_ml_service nai1ka/predict_taxi_price_ml_service
# Put image to Docker Hun
sudo docker push nai1ka/predict_taxi_price_ml_service:latest
# Run the container
sudo docker run --rm -p 5151:8080 predict_taxi_price_ml_service -d
