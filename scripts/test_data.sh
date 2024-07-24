#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e

cd $PYTHONPATH

# Step 1: Take a data sample
echo "Taking a data sample..."
python3 sample_data.py
echo "Data sample taken"


# Step 2: Validate the data sample
echo "Validating the data sample..."
VALIDATION_OUTPUT=$(python3 validate_data.py)

if echo "$VALIDATION_OUTPUT" | tail -1 | grep -q "All data validations passed."; then
    VALIDATION_STATUS="valid"
else
    VALIDATION_STATUS="invalid"
fi

if [ "$VALIDATION_STATUS" = "valid" ]; then
    echo "Data sample validated successfully"
    
    cd ../scripts

    # Step 3: Version data
    echo "Versioning data..."
    bash version_data.sh
    echo "Data versioned"
else
    echo "Data validation failed. Data will not be versioned."
    exit 1
fi