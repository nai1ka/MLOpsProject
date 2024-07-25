#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e

cd $PWD/src

# Step 1: Take a data sample
echo "Taking a data sample..."
python3 sample_data.py
echo "Data sample taken"


# Step 2: Validate the data sample
echo "Validating the data sample..."
python3 validate_data.py
echo "Data sample validated"

cd ../scripts
# Step 3: Version data
bash version_data.sh