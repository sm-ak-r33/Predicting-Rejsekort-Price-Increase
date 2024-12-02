import os
import warnings
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prophet import Prophet 
import mlflow
from mlflow.models.signature import infer_signature
import dagshub
from urllib.parse import urlparse

# Initialize DagsHub
dagshub.init(repo_owner='smahasanulkarim', repo_name='Predicting-Rejsekort-Price-Increase-2023', mlflow=True)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the rejsekort CSV file from the URL
    cleaned_data_url = (
        "https://github.com/sm-ak-r33/Predicting-Rejsekort-Price-Increase-2023/raw/refs/heads/main/data_cleaned.csv"
    )
    try:
        df = pd.read_csv(cleaned_data_url)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
        sys.exit(1)

    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        r2 = r2_score(actual, pred)
        return rmse, mae, mape, r2

    # Logging with MLflow
    with mlflow.start_run():
        # Ensure the date column is in datetime format
        df['date'] = pd.to_datetime(df['date'])

        # Filter rows with dates from 1st January 2021 onwards
        df = df[df['date'] >= '2021-01-01'].sort_values(by='date')

        # Prepare the data for Prophet (renaming columns)
        df_prophet = df[['date', 'passengers']].rename(columns={'date': 'ds', 'passengers': 'y'})

        # Train-test split (80:20)
        split_index = int(len(df_prophet) * 0.8)
        train_prophet = df_prophet.iloc[:split_index]
        test_prophet = df_prophet.iloc[split_index:]

        # Fit the Prophet model
        prophet_model = Prophet()
        prophet_model.fit(train_prophet)

        # Create a dataframe with future dates for forecasting
        future = prophet_model.make_future_dataframe(periods=len(test_prophet), freq='D')

        # Forecast the test set period
        forecast = prophet_model.predict(future)
        forecast = forecast[['ds', 'yhat']].set_index('ds')

        # Align the predictions with the test data
        test_dates = test_prophet['ds']
        test_y = test_prophet['y']
        predictions = forecast.loc[test_dates, 'yhat']

        # Evaluate metrics
        rmse, mae, mape, r2 = eval_metrics(test_y.values, predictions.values)

        # Log evaluation metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("r2", r2)

        # Print metrics
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  MAPE: {mape}")
        print(f"  R2: {r2}")

        # Set tracking URI for DagsHub
        dagshub_tracking_uri = "https://dagshub.com/smahasanulkarim/Predicting-Rejsekort-Price-Increase-2023.mlflow"
        mlflow.set_tracking_uri(dagshub_tracking_uri)

        # Log the model
        mlflow.prophet.log_model(prophet_model, "model", registered_model_name="Prophet")
