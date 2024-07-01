#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e

cd ../src

# Step 1: Take a data sample
echo "Taking a data sample..."
python3 sample_data.py
echo "Data sample taken"


# Step 2: Validate the data sample
echo "Validating the data sample..."
python3 validate_data.py
echo "Data sample validated"


# Step 3: Version the data sample using DVC
echo "Versioning the data sample..."
version=$(cat ../configs/main.yaml | shyaml get-value data.sample_version)
echo "Data sample version: v$version"
dvc add ../data/samples/sample.csv
git add ../data/samples/sample.csv.dvc
git commit -m "Add and version data sample"
git push origin main
git tag -a "v$version" -m "add data version v$version"
git push --tags
dvc push
echo "Data sample versioned"

echo "Process completed successfully"