from prefect import flow, task
import pandas as pd
import torch
import mlflow
from mlflow.tracking import MlflowClient
from data_collection import load_data
from data_processing import convert_to_timeseries
from data_processing import resample_timeseries
from data_processing import prepare_X_y
from train import create_dataloader
from train import train_model
from train import save_model, register_model
from plotting import plot_learning_curve
from model_registry import register_model
from model_registry import transition_model

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

@task
def save_model_locally(model, model_dir:str,
                       model_format:str):
    save_model(model, model_dir, model_format)



@task
def mlflow_registry(client, run, model_name:str):
    
    register_model(run.info.run_id, model_name)
    print('Model registered in MLflow')
      
    latest_versions = client.get_latest_versions(name = model_name)
    version = latest_versions[-1].version
    stage = "Production"
    transition_model(client, model_name, version, stage)

    print(f'The model version {version} was transitioned to {stage}')


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
                      MLFLOW_EXPERIMENT_NAME:str = "fuel-price-experiment",
                      model_dir:str = "models",
                      model_format:str = "pickle",
                      model_name:str = "fuel-price-predictor"):
    
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
    client = MlflowClient(tracking_uri = MLFLOW_TRACKING_URI)

    model, run, train_hist, val_hist = train_torch_model(train_loader, 
                                                         val_loader, 
                                                         num_epochs, 
                                                         learning_rate)
    

    save_model_locally(model, model_dir, model_format)

    print(f'Current MLflow run id: {run.info.run_id}')
    mlflow_registry(client, run,  model_name)

    
    plot_learning_curve(num_epochs, train_hist, val_hist)

if __name__ == "__main__":
    ml_pipeline()

    
