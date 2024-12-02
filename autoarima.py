import os
import warnings
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import dagshub
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from pmdarima import auto_arima

dagshub.init(repo_owner='smahasanulkarim', repo_name='Predicting-Rejsekort-Price-Increase-2023', mlflow=True)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

df = pd.read_csv('')

def eval_metrics(actual, pred):

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    r2 = r2_score(actual, pred)

    return rmse, mae,mape, r2 
 
# Parameters for auto_arima
seasonal = True  # Adjust based on your data
m = 7  # Set to seasonal period (e.g., 12 for monthly data if seasonal=True)
start_p = 0
start_q = 0
max_p = 5
max_q = 5

# Logging with MLflow
with mlflow.start_run():

    # Ensure the date column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Filter rows with dates from 1st January 2021 onwards
    df = df[df['date'] >= '2021-01-01'].sort_values(by='date')

    # Ensure the time series is indexed by the date column
    df.set_index('date', inplace=True)

    # Extract the series
    series = df['passengers']

    # Train-test split (80:20)
    split_index = int(len(series) * 0.8)
    train, test = series[:split_index], series[split_index:]
    arima_model = auto_arima(
        train,   
        seasonal=seasonal,
        m=m,
        start_p=start_p,
        start_q=start_q,
        max_p=max_p,
        max_q=max_q,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    # Make predictions
    predicted_qualities = arima_model.predict(n_periods=len(test))

    # Evaluate metrics
    (rmse, mae, mape, r2) = eval_metrics(test, predicted_qualities)

    # Log parameters specific to auto_arima
    mlflow.log_param("seasonal", seasonal)
    mlflow.log_param("m", m)
    mlflow.log_param("start_p", start_p)
    mlflow.log_param("start_q", start_q)
    mlflow.log_param("max_p", max_p)
    mlflow.log_param("max_q", max_q)

    # Log evaluation metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("r2", r2)

    # Print metrics
    print(f"Auto-ARIMA model (seasonal={seasonal}, m={m}):")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  MAPE: {mae}")
    print(f"  R2: {r2}")

    dagshub_tracking_uri = "https://dagshub.com/smahasanulkarim/Predicting-Rejsekort-Price-Increase-2023.mlflow"
    mlflow.set_tracking_uri(dagshub_tracking_uri)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(
            arima_model, "model", registered_model_name="AutoArima")
    else:
        mlflow.sklearn.log_model(arima_model, "model")