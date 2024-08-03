from prefect import flow, task
from data_collection import load_data
from data_processing import convert_to_timeseries
from data_processing import resample_timeseries
from data_processing import prepare_X_y
from train import create_dataloader, train_model
from plotting import plot_learning_curve
import pandas as pd
import torch
import mlflow

@task
def collect_data(root_directory: str, station_uuid: str):
    df = load_data(root_directory, station_uuid)
    print(f'The dataset has {len(df)} samples.')
    return df

@task
def preprocess_data(df: pd.DataFrame, 
                    col_name: str, 
                    sampling_freq: str):
    df = convert_to_timeseries(df)
    df_resampled = resample_timeseries(df, col_name, sampling_freq)
    return df_resampled

@task
def split_data(df: pd.DataFrame, train_endpoint: str,
               val_startpoint: str, val_endpoint: str,
               test_startpoint: str):
    df_train = df[:train_endpoint]
    df_val = df[val_startpoint: val_endpoint]
    df_test = df[test_startpoint: ]
    return df_train, df_val, df_test

@task
def prepare_X_y_train_val(df_train: pd.DataFrame, df_val: pd.DataFrame):
    X_train, y_train = prepare_X_y('training', df_train, sequence_length=48)
    X_val, y_val = prepare_X_y('training', df_val, sequence_length=24)

    return X_train, y_train, X_val, y_val

@task
def create_torch_dataloader(X_train: torch.float32, y_train: torch.float32,
                            X_val:  torch.float32, y_val: torch.float32,
                            batch_size: int = 16):
    
    train_loader, val_loader = create_dataloader(X_train, y_train, 
                                                 X_val, y_val, batch_size)
    
    return train_loader, val_loader

@task
def train_torch_model(train_loader,val_loader,
                      num_epochs: int, learning_rate:float):
    model, run, train_hist, val_hist = train_model(train_loader,val_loader, 
                                       num_epochs, learning_rate)
    
    return model, run, train_hist, val_hist


@flow(log_prints= True)  
def ml_pipeline(root_directory: str = 'C:/BITrusted/fuel-price-prediction-mlops/data/2024_prices',
                      station_uuid: str = '28d2efc8-a5e6-47d6-9d37-230fbcefcf70',
                      train_endpoint: str = '2024-05-31',
                      val_startpoint: str = '2024-06-01', 
                      val_endpoint: str = '2024-07-13',
                      test_startpoint: str = '2024-07-14',
                      fuel_type: str = 'e5', 
                      sampling_freq: str = '1h',
                      num_epochs:int = 50,
                      learning_rate:float = 1e-3,
                      MLFLOW_TRACKING_URI:str = "sqlite:///mlflow.db",
                      MLFLOW_EXPERIMENT_NAME:str = "fuel-price-experiment"):
    
    df = collect_data(root_directory, station_uuid)

    df_processed = preprocess_data(df, fuel_type, sampling_freq)

    df_train, df_val, df_test = split_data(df_processed, 
                                           train_endpoint,
                                           val_startpoint, val_endpoint,
                                           test_startpoint)
    
    X_train, y_train, X_val, y_val = prepare_X_y_train_val(df_train, df_val)

    train_loader, val_loader = create_torch_dataloader(X_train, y_train,
                                                       X_val, y_val)
        
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    model, run, train_hist, val_hist = train_torch_model(train_loader, 
                                                         val_loader, 
                                                         num_epochs, 
                                                         learning_rate)
    print(f'Current MLflow run id: {run.info.run_id}')
    plot_learning_curve(num_epochs, train_hist, val_hist)

if __name__ == "__main__":
    ml_pipeline()

    
