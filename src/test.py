import os
import zenml
from zenml.client import Client
import dvc.api

BASE_PATH = os.path.expandvars("$PROJECTPATH")
print(BASE_PATH)
t = dvc.api.get_url(
        rev = "v2",
        path = "data/samples/sample.csv",
        remote = "localstore",
        repo = BASE_PATH
    )
print(t)