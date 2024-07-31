import numpy as np
import mlflow
import pandas as pd
from data_processing import prepare_X_y
from data_processing import generate_future_timestamps
from plotting import plot_forecast



def forecast(val_data, model):

    X_val, _ = prepare_X_y('validation', val_data, sequence_length=24)

    # Define the number of future time steps to forecast
    num_forecast_steps = 48 # next timesteps to forecast
    time_interval_min = 60 # @minute intervals

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

        # Update the historical_data sequence by removing the oldest value and adding the predicted value
        historical_data = np.roll(historical_data, shift=-1)
        historical_data[-1] = predicted_value.values[0]
 
          
    # Generate futute dates
    last_timestamp = val_data.index[-1]

    # Generate the next stipulated timepoints
    future_timestamps = generate_future_timestamps(last_timestamp, num_forecast_steps, time_interval_min)

    # Concatenate the original index with the future dates
    combined_index = val_data.index.append(future_timestamps)

    return prediction_horizon, future_timestamps, forecasted_values, combined_index, sequence_to_plot



if __name__ == '__main__':

    data_path = '../data/2024_globus_gas_prices.parquet'
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    data = df['e5']

    val_data_path = '../data/2024_val_data.parquet'
    val_data = pd.read_parquet(val_data_path)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fuel-price-experiment")

    # Load model as a PyFuncModel
    logged_model = 'runs:/f83b41f1ac2b450893bbcdaac0e5d157/LSTM-model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    prediction_horizon, future_timestamps, forecasted_values, combined_index, sequence_to_plot = forecast(val_data, 
                                                                                                      loaded_model)

    plot_forecast(data, val_data, 
                  prediction_horizon, future_timestamps, 
                  forecasted_values, combined_index, 
                  sequence_to_plot)
        
   