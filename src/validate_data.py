import sys

import hydra
import pandas as pd
from great_expectations.data_context import FileDataContext
from omegaconf import DictConfig


def validate_data(cfg: DictConfig = None):
    """
    Validate the initial data using Great Expectations.
    
    This function initializes a Great Expectations context, sets up a data source,
    creates a batch request, and validates the data against a predefined expectation suite.
    
    Args:
        cfg (DictConfig): Configuration object from Hydra.
    """
    context = FileDataContext(context_root_dir="../services/gx")

    # Load the expectation suite
    suite_name = "initial_data_validation"

    # Add a pandas datasource (if not already added)
    ds = context.sources.add_or_update_pandas(name="pandas_datasource")

    # Add a CSV asset for the new data
    sample = ds.add_csv_asset(
        name="sample_csv",
        filepath_or_buffer=f"../{cfg.data_path}"
    )

    # Build a batch request
    batch_request = sample.build_batch_request()

    # Get a validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    # Validate the data
    results = validator.validate()

    if not results.success:
        failed_expectations = [
            (result.expectation_config.expectation_type, result.result)
            for result in results.results
            if not result.success
        ]
        raise AssertionError(f"Data validation failed: {failed_expectations}")

    print("All data validations passed.")


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def validate_initial_data(cfg=None):
    validate_data(cfg)


if __name__ == "__main__":
    validate_initial_data()
