import hydra
from omegaconf import DictConfig
import pandas as pd
from great_expectations.data_context import FileDataContext


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def sample_data(cfg: DictConfig = None):
    """Create a sample of the dataset"""
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


def validate_initial_data():
    """Validate the initial data using Great Expectations"""
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
    # id: unique, not null, pattern of UUID
    validator.expect_column_values_to_not_be_null("id")
    validator.expect_column_values_to_be_unique("id")
    validator.expect_column_values_to_match_regex("id",
                                                  r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$")

    # price: not null, >0, is of type float
    validator.expect_column_values_to_be_between("price", min_value=0, max_value=None, strict_min=True)
    validator.expect_column_values_to_be_of_type("price", "float")

    # distance: not null, >0, is of type float
    validator.expect_column_values_to_not_be_null("distance")
    validator.expect_column_values_to_be_between("distance", min_value=0, max_value=None, strict_min=True)
    validator.expect_column_values_to_be_of_type("distance", "float")

    # datetime: not null, format, min value
    validator.expect_column_values_to_not_be_null("datetime")
    validator.expect_column_values_to_match_strftime_format("datetime", "%Y-%m-%d %H:%M:%S")
    validator.expect_column_values_to_be_between("datetime", min_value="2018-11-26 00:00:00", max_value=None)

    # hour: not null, 0 to 24, is of type int
    validator.expect_column_values_to_not_be_null("hour")
    validator.expect_column_values_to_be_between("hour", 0, 24)
    validator.expect_column_values_to_be_of_type("hour", "int")

    # day: not null, 1 to 31, is of type int
    validator.expect_column_values_to_not_be_null("day")
    validator.expect_column_values_to_be_between("day", 1, 31)
    validator.expect_column_values_to_be_of_type("day", "int")

    # month: not null, 1 to 12, is of type int
    validator.expect_column_values_to_not_be_null("month")
    validator.expect_column_values_to_be_between("month", 1, 12)
    validator.expect_column_values_to_be_of_type("month", "int")

    # cab_type: not null, in set of values {Lyft, Uber}
    validator.expect_column_values_to_be_in_set("cab_type", {"Lyft", "Uber"})
    validator.expect_column_values_to_not_be_null("cab_type")

    results = validator.validate()

    if not results.success:
        failed_expectations = [
            (result.expectation_config.expectation_type, result.result)
            for result in results.results
            if not result.success
        ]
        raise AssertionError(f"Data validation failed: {failed_expectations}")

    print("All data validations passed.")
