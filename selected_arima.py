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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
 

dagshub.init(repo_owner='smahasanulkarim', repo_name='Predicting-Rejsekort-Price-Increase-2023', mlflow=True)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the rejsekort csv file from the URL
    cleaned_data_url = (
        "https://github.com/sm-ak-r33/Predicting-Rejsekort-Price-Increase-2023/raw/refs/heads/main/data_cleaned.csv"
    )
    try:
        df = pd.read_csv(cleaned_data_url)
    except Exception as e:
        logger.exception(
            "Unable to download training & test excel, check your internet connection. Error: %s", e
        )


def eval_metrics(actual, pred):

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    r2 = r2_score(actual, pred)

    return rmse, mae,mape, r2 


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
    
    # Define SARIMA model with seasonal differencing
    sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    sarima_fit = sarima_model.fit(disp=False)
    

    predicted_qualities = sarima_fit.forecast(steps=len(test))

    # Evaluate metrics
    (rmse, mae, mape, r2) = eval_metrics(test, predicted_qualities)

    # Log parameters specific to auto_arima
    mlflow.log_param("seasonal", True)
    mlflow.log_param("m", 7)
    mlflow.log_param("p", 1)
    mlflow.log_param("d", 1)
    mlflow.log_param("q", 1)
    mlflow.log_param("P", 1)
    mlflow.log_param("D", 1)
    mlflow.log_param("Q", 1)
    
    # Log evaluation metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("r2", r2)

    # Print metrics
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
            sarima_model, "model", registered_model_name="SARIMA(1,1,1)(1,1,1,7)")
    else:
        mlflow.sklearn.log_model(sarima_model, "model")