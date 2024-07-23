#!/bin/bash

# Ensure the script exits on error
set -e

# Define an array with the example versions
example_versions=("v1" "v2" "v3" "v4" "v5")

# Define the port and random state to be used for the predictions
port=5151
random_state=14

# Loop through each version and run the MLflow predict command
for version in "${example_versions[@]}"; do
    echo "Testing prediction for example_version=${version}"
    mlflow run . --env-manager local -e predict -P example_version=${version} -P port=${port} -P random_state=${random_state}
done

echo "All samples tested successfully."