import hydra
from omegaconf import DictConfig
import pandas as pd
import os


# TODO add comments


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def sample_data(cfg: DictConfig = None):
    dataset_name = cfg.data.dataset_name
    output_name = cfg.data.output_name

    df = pd.read_csv("../data/" + dataset_name)
    sample_size = int(cfg.data.sample_size * len(df))
    sample_df = df[0:sample_size]
    sample_df.to_csv("../data/samples/"+output_name, index=False)


from great_expectations.data_context import FileDataContext

def validate_initial_data():
    context = FileDataContext(context_root_dir="../services/gx")
    ds = context.sources.add_or_update_pandas(name="pandas_datasource")
    da1 = ds.add_csv_asset(
        name="csv_file",
        filepath_or_buffer="../data/samples/sample.csv"
    )

    batch_request = da1.build_batch_request()

    context.add_or_update_expectation_suite("initial_data_validation")
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="initial_data_validation"
    )

    ex1 = validator.expect_column_values_to_be_unique(column="id")
    assert ex1['success']


if __name__ == "__main__":
    sample_data()
    #validate_initial_data()
