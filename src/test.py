import mlflow
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,mean_absolute_percentage_error

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import zenml
from zenml.client import Client

import os
import zenml
from zenml.client import Client
import dvc.api

from data import transform_data
from evaluate import load_local_model
BASE_PATH = os.path.expandvars ("$PROJECTPATH")


test_data = pd.read_csv(BASE_PATH+"/data/rideshare_kaggle.csv")
model: mlflow.pyfunc.PyFuncModel = load_local_model("challenger")
#mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
X,y = transform_data(
            df = test_data, 
            cfg = None,
            version = "v2", 
            return_df = False
        )

#linear_model = DecisionTreeRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#linear_model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)


# Print the metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")
print(f'MAPE: {mape}')