# pylint: disable=duplicate-code
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request

# Correcting the typo from os.path.json to os.path.join
sys.path.append(os.path.abspath(os.path.join("..", "src")))

# pylint: disable=wrong-import-position
from data_processing import prepare_X_y
from plotting import plot_forecast_web_service
from utils import generate_future_timestamps


def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]

    return obj


def forecast(model, recent_data, num_forecast_steps):

    X_val, _ = prepare_X_y("recent data", recent_data, sequence_length=24)

    # Convert to NumPy and remove singleton dimensions
    sequence_to_plot = X_val.squeeze().cpu().numpy()

    # Use the last 24 data points as the starting point
    historical_data = sequence_to_plot[-1]

    # Initialize a list to store the forecasted values
    forecasted_values = []

    prediction_horizon = num_forecast_steps + len(historical_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for _ in range(prediction_horizon):
            # Prepare the historical_data tensor
            historical_data_tensor = (
                torch.as_tensor(historical_data).view(1, -1, 1).float().to(device)
            )
            # Use the model to predict the next value
            predicted_value = model(historical_data_tensor).cpu().numpy()[0, 0]

            # Append the predicted value to the forecasted_values list
            forecasted_values.append(predicted_value[0])

            # Update the historical_data sequence
            # remove the oldest value and add the predicted value
            historical_data = np.roll(historical_data, shift=-1)

            if torch.is_tensor(predicted_value):
                predicted_value = predicted_value.cpu().numpy()
            if torch.is_tensor(historical_data):
                historical_data = historical_data.cpu().numpy()

            historical_data[-1] = predicted_value.item()

    return prediction_horizon, forecasted_values


def predict(recent_data, num_forecast_steps):

    model_docker_path = "fuel_price_lstm.pickle"
    with open(model_docker_path, "rb") as f_in:
        loaded_model = pickle.load(f_in)

    prediction_horizon, forecasted_values = forecast(
        loaded_model,
        recent_data,
        num_forecast_steps,
    )
    return prediction_horizon, forecasted_values


def plot_prediction(
    recent_data, num_forecast_steps, prediction_horizon, forecasted_values
):
    # Generate futute dates
    last_timestamp = recent_data.index[-1]

    # Define the number of future time steps to forecast
    time_interval_min = 60  # @minute intervals

    # Generate the next stipulated timepoints
    future_timestamps = generate_future_timestamps(
        last_timestamp,
        num_forecast_steps,
        time_interval_min,
    )

    # Concatenate the original index with the future dates
    combined_index = recent_data.index.append(future_timestamps)

    plot_forecast_web_service(
        recent_data, prediction_horizon, forecasted_values, combined_index
    )


app = Flask("fuel-price-prediction")


@app.route("/predict", methods=["POST"])
def forecast_endpoint():
    data = request.get_json()

    recent_data = pd.DataFrame(data["recent_data"])
    num_forecast_steps = data["num_forecast_steps"]

    pred = predict(recent_data, num_forecast_steps)

    result = {"predicted_steps": pred[0], "forecasted_prices": pred[1]}

    serializable_result = convert_to_serializable(result)

    return jsonify(serializable_result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
