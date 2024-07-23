import mlflow
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,mean_absolute_percentage_error

import zenml
from zenml.client import Client



import os
import zenml
from zenml.client import Client
import dvc.api

from data import extract_data, transform_data
from evaluate import load_local_model
BASE_PATH = os.path.expandvars ("$PROJECTPATH")


test_data = pd.read_csv(BASE_PATH+"/data/samples/sample.csv")
#model: mlflow.pyfunc.PyFuncModel = load_local_model("challenger")
mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
X,y = transform_data(
            df = test_data, 
            cfg = None,
            version = "v2", 
            return_df = False
        )
print(X)
y_pred = model.predict(X)
print(y_pred)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
mape = mean_absolute_percentage_error(y, y_pred)


# Print the metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")
print(f'MAPE: {mape}')