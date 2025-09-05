import pandas as pd
import numpy as np

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

from urllib.parse import urlparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

apple_experiment = mlflow.set_experiment("Apple_Models")
run_name = "apples_rf_test"
artifact_path = "rf_apples"


data = pd.read_csv("data/apple_sales_data.csv")

X = data.drop(columns=["date", "demand"])
y = data["demand"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

params = {
    "n_estimators": 150,
    "max_depth": 8,
    "min_samples_split": 15,
    "min_samples_leaf": 6,
    "bootstrap": True,
    "oob_score": False,
    "random_state": 888,
}

with mlflow.start_run(run_name=run_name):

    mlflow.log_params(params)
    
    rf = RandomForestRegressor(**params)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    signature = infer_signature(X_train, y_train)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    mlflow.log_metrics({"mae": mae, "mse": mse, "rmse": rmse, "r2": r2})

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    
    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(rf, "model", signature=signature, registered_model_name=artifact_path)
    else:
        mlflow.sklearn.log_model(rf, "model", signature=signature)