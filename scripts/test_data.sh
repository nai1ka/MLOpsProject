#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e

cd ../src

# Step 1: Take a data sample
echo "Taking a data sample..."
python -c "
from data import sample_data
from hydra import compose, initialize

with initialize(config_path=\"../configs\", version_base=None):
  cfg = compose(config_name=\"main\")
  sample_data(cfg)
"
echo "Data sample taken"


# Step 2: Validate the data sample
echo "Validating the data sample..."
python -c "
from data import validate_initial_data
try:
    validate_initial_data()
    print('Data validation passed.')
except Exception as e:
    print(f'Data validation failed: {e}')
    exit(1)
"
echo "Data sample validated"

# Step 3: Version the data sample using DVC
echo "Versioning the data sample..."
dvc add ../data/samples/sample.csv
git add ../data/samples/sample.csv.dvc
git commit -m "Add and version data sample"
dvc push
git push
echo "Data sample versioned"

echo "Process completed successfully"