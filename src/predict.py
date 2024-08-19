import mlflow
import numpy as np
import pandas as pd
import torch

from data_processing import prepare_X_y
from plotting import plot_forecast
from utils import generate_future_timestamps


def forecast_future_values_local(model, historical_data, num_forecast_steps, device):
    forecasted_values = []
    prediction_horizon = num_forecast_steps + len(historical_data)

    with torch.no_grad():
        for _ in range(prediction_horizon):
            historical_data_tensor = (
                torch.as_tensor(historical_data).view(1, -1, 1).float().to(device)
            )
            predicted_value = model(historical_data_tensor).cpu().numpy()[0, 0]
            forecasted_values.append(predicted_value[0])

            historical_data = np.roll(historical_data, shift=-1)
            historical_data[-1] = predicted_value

    return prediction_horizon, forecasted_values


def local_model_forecast(model, data_name, data, num_forecast_steps):

    X_val, _ = prepare_X_y(data_name, data, sequence_length=24)

    # Define the number of future time steps to forecast
    time_interval_min = 60  # @minute intervals

    # Convert to NumPy and remove singleton dimensions
    sequence_to_plot = X_val.squeeze().cpu().numpy()

    # Use the last 24 data points as the starting point
    historical_data = sequence_to_plot[-1]

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Forecast values
    prediction_horizon, forecasted_values = forecast_future_values_local(
        model, historical_data, num_forecast_steps, device
    )

    # Generate futute dates
    last_timestamp = data.index[-1]

    # Generate the next stipulated timepoints
    future_timestamps = generate_future_timestamps(
        last_timestamp, num_forecast_steps, time_interval_min
    )

    # Concatenate the original index with the future dates
    combined_index = data.index.append(future_timestamps)

    return (
        prediction_horizon,
        future_timestamps,
        forecasted_values,
        combined_index,
        sequence_to_plot,
    )


def forecast(model, data_name, data, num_forecast_steps):

    X_val, _ = prepare_X_y(data_name, data, sequence_length=24)

    # Define the number of future time steps to forecast
    time_interval_min = 60  # @minute intervals

    # Convert to NumPy and remove singleton dimensions
    sequence_to_plot = X_val.squeeze().cpu().numpy()

    # Use the last 24 data points as the starting point
    historical_data = sequence_to_plot[-1]

    # Initialize a list to store the forecasted values
    forecasted_values = []

    prediction_horizon = num_forecast_steps + len(historical_data)

    # Use the trained model to forecast future values
    for _ in range(prediction_horizon):

        # Use the model to predict the next value
        predicted_value = model.predict(pd.DataFrame(historical_data))

        # Append the predicted value to the forecasted_values list
        forecasted_values.append(predicted_value.values[0][0])

        # Update the historical_data sequence
        # remove the oldest value and add the predicted value
        historical_data = np.roll(historical_data, shift=-1)
        historical_data[-1] = predicted_value.values[0]

    # Generate futute dates
    last_timestamp = data.index[-1]

    # Generate the next stipulated timepoints
    future_timestamps = generate_future_timestamps(
        last_timestamp, num_forecast_steps, time_interval_min
    )

    # Concatenate the original index with the future dates
    combined_index = data.index.append(future_timestamps)

    return (
        prediction_horizon,
        future_timestamps,
        forecasted_values,
        combined_index,
        sequence_to_plot,
    )


if __name__ == '__main__':

    data_path = 'data/2024_globus_gas_prices.parquet'
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    refrence_data = df['e5']

    val_data_path = 'data/2024_val_data.parquet'
    val_data = pd.read_parquet(val_data_path)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fuel-price-experiment")

    # Load model as a PyFuncModel
    run_id = '2f41d5e9dc5f4e15a707755bef4386b5'
    logged_model = f'runs:/{run_id}/LSTM-model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    dataName = 'validation'
    forecast_steps = 24
    horizon, forecast_timestamps, predictions, indices, plot_sequence = forecast(
        loaded_model,
        dataName,
        val_data,
        forecast_steps,
    )
    forecast_params = {
        'prediction_horizon': horizon,
        'future_timestamps': forecast_timestamps,
        'forecasted_values': predictions,
        'combined_index': indices,
        'sequence_to_plot': plot_sequence,
    }

    plot_forecast(refrence_data, val_data, forecast_params)
