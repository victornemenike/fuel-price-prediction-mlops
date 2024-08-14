import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_forecast(data, val_data, forecast_params):
    prediction_horizon = forecast_params['prediction_horizon']
    future_timestamps = forecast_params['future_timestamps']
    forecasted_values = forecast_params['forecasted_values']
    combined_index = forecast_params['combined_index']
    sequence_to_plot = forecast_params['sequence_to_plot']

    # set the size of the plot
    _, ax = plt.subplots(figsize=(12, 4))

    # Test data
    ax.plot(val_data.index[-100:-24], val_data[-100:-24], label="val_data", color="b")
    # reverse the scaling transformation
    original_cases = np.expand_dims(sequence_to_plot[-1], axis=0).flatten()

    # the historical data used as input for forecasting
    ax.plot(val_data.index[-24:], original_cases, label='actual values', color='green')

    # reverse the scaling transformation
    forecasted_cases = np.expand_dims(forecasted_values, axis=0).flatten()
    # plotting the forecasted values
    ax.plot(
        combined_index[-prediction_horizon:],
        forecasted_cases,
        label='forecasted values',
        color='red',
    )

    ds = data.index
    test_timestamps = ds[(ds >= future_timestamps[0]) & (ds <= future_timestamps[-1])]

    ax.plot(
        test_timestamps,
        data[future_timestamps[0] : future_timestamps[-1]],
        label='test data (snippet)',
        color='black',
    )

    plt.xlabel('Time')
    plt.ylabel('Price (€)')
    plt.legend()
    plt.title('Petrol Price Forecasting')
    plt.grid(True)
    plt.show()


def plot_forecast_web_service(
    recent_data, prediction_horizon, forecasted_values, combined_index
):

    _, ax = plt.subplots(figsize=(12, 4))

    # plot recent data
    ax.plot(recent_data.index, recent_data, label="historical data", color="b")

    # plot forecasted values
    forecasted_cases = np.expand_dims(forecasted_values, axis=0).flatten()
    ax.plot(
        combined_index[-prediction_horizon:],
        forecasted_cases,
        label='forecasted',
        color='red',
    )

    plt.xlabel('Time')
    plt.ylabel('Price (€)')
    plt.legend()
    plt.title('Petrol Price Forecasting')
    plt.grid(True)
    plt.show()


def predict_actual_dist(df, future_timestamps, forecasted_cases):
    warnings.filterwarnings("ignore", category=UserWarning)
    sns.distplot(forecasted_cases, label='prediction')
    sns.distplot(df[future_timestamps[0] : future_timestamps[-1]], label='actual')
    plt.legend()
    plt.show()


def plot_learning_curve(num_epochs, train_hist, val_hist):
    _, ax = plt.subplots(figsize=(12, 4))
    x = np.linspace(1, num_epochs, num_epochs)
    ax.plot(x, train_hist, scalex=True, label="Training loss")
    ax.plot(x, val_hist, label="Validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
