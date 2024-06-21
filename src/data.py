import hydra
from omegaconf import DictConfig
import pandas as pd


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def sample_data(cfg: DictConfig = None):
    dataset_name = cfg.data.dataset_name
    output_name = cfg.data.output_name

    # Load the dataset
    df = pd.read_csv("../data/" + dataset_name)

    # Calculate the sample size
    sample_size = int(cfg.data.sample_size * len(df))

    # Create the sample
    sample_df = df[0:sample_size]

    # Save the sample
    sample_df.to_csv("../data/samples/" + output_name, index=False)


from great_expectations.data_context import FileDataContext


def validate_initial_data():
    # Create a data context
    context = FileDataContext(context_root_dir="../services/gx")

    # Add a pandas datasource
    ds = context.sources.add_or_update_pandas(name="pandas_datasource")

    # Add a CSV asset
    da1 = ds.add_csv_asset(
        name="csv_file",
        filepath_or_buffer="../data/samples/sample.csv"
    )

    # Build a batch request
    batch_request = da1.build_batch_request()

    # Create an expectation suite
    context.add_or_update_expectation_suite("initial_data_validation")
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="initial_data_validation"
    )

    # Validations
    ex1 = validator.expect_column_values_to_be_unique(column="id")
    assert ex1['success']
