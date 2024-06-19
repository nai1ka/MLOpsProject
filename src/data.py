import hydra
from omegaconf import DictConfig
import pandas as pd
import os


# TODO add comments


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def sample_data(cfg: DictConfig = None):
    os.environ['KAGGLE_USERNAME'] = 'nai1ka'
    os.environ['KAGGLE_KEY'] = 'a44922ed88b8f8675c22ef8269d2232c'

    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    data_url = cfg.data.url
    sample_size = cfg.data.sample_size
    dataset_name = cfg.data.dataset_name

    kaggle.api.dataset_download_files(data_url, path="../data/temp", unzip=True)

    df = pd.read_csv("../data/temp/" + dataset_name)
    sample_df = df.sample(frac=sample_size, random_state=1)
    sample_df.to_csv("../data/samples/sample.csv", index=False)

    # Clean up temporary directory
    os.remove("../data/temp/" + dataset_name)
    os.rmdir("../data/temp")


from great_expectations.data_context import FileDataContext


def validate_initial_data():
    context = FileDataContext(context_root_dir="../services/gx")
    ds = context.sources.add_or_update_pandas(name="pandas_datasource")
    da1 = ds.add_csv_asset(
        name="csv_file",
        filepath_or_buffer="../data/samples/sample.csv"
    )

    batch_request = da1.build_batch_request()
    batches = da1.get_batch_list_from_batch_request(batch_request)
    for batch in batches:
        print(batch.batch_spec)
    context.add_or_update_expectation_suite("initial_data_validation")
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="initial_data_validation"
    )

    ex1 = validator.expect_column_values_to_be_unique(column="id")
    assert ex1['success']


if __name__ == "__main__":
    # sample_data()
    validate_initial_data()
