import os
import warnings
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature
import dagshub

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

    # Logging with MLflow
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        # Parameters
        look_back = 30
        epochs = 75
        batch_size = 32

        # Preprocessing
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= '2021-01-01'].sort_values(by='date')

        value_col = "passengers"
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[[value_col]])

        # Create sequences
        def create_sequences(data, look_back):
            X, y = [], []
            for i in range(len(data) - look_back):
                X.append(data[i:i + look_back])
                y.append(data[i + look_back])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data, look_back)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Reshape for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Build the model
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(look_back, 1)),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mape')

        # Log custom parameters
        mlflow.log_param("look_back", look_back)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        # Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )

        # Predictions
        y_pred = model.predict(X_test).flatten()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Evaluation Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("r2", r2)

        # Set tracking URI for DagsHub
        dagshub_tracking_uri = "https://dagshub.com/smahasanulkarim/Predicting-Rejsekort-Price-Increase-2023.mlflow"
        mlflow.set_tracking_uri(dagshub_tracking_uri)

        # Specify the model file format (keras_model_kwargs)
        keras_model_kwargs = {"save_format": "h5"} 
        mlflow.tensorflow.log_model(model, "model", keras_model_kwargs=keras_model_kwargs)

        # Print metrics
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  MAPE: {mape}")
        print(f"  R2: {r2}")
