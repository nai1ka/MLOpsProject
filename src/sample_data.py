import hydra
from omegaconf import DictConfig
import pandas as pd
from great_expectations.data_context import FileDataContext
import sys


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def sample_data(cfg: DictConfig = None):
    """Create a sample of the dataset"""
    dataset_name = cfg.data.dataset_name
    output_name = cfg.data.output_name
    sample_version = cfg.data.sample_version

    # Load the dataset
    df = pd.read_csv("../data/" + dataset_name)

    # Calculate the sample size
    sample_size = int(cfg.data.sample_size * len(df))

    # Create the sample
    sample_df = df[(sample_version - 1) * sample_size:sample_version * sample_size]

    # Save the sample
    sample_df.to_csv("../data/samples/" + output_name, index=False)


def validate_initial_data():
    """Validate the initial data using Great Expectations"""
    context = FileDataContext(context_root_dir="../services/gx")

    # Load the expectation suite
    suite_name = "initial_data_validation"

    # Add a pandas datasource (if not already added)
    ds = context.sources.add_or_update_pandas(name="pandas_datasource")

    # Add a CSV asset for the new data
    sample = ds.add_csv_asset(
        name="sample_csv",
        filepath_or_buffer="../data/samples/sample.csv"
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        match sys.argv[1]:
            case "sample":
                sample_data()
            case "validate":
                validate_initial_data()
            case _:
                raise ValueError("Invalid argument")
