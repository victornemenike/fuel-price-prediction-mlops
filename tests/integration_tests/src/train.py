# pylint: disable=duplicate-code
# pylint: disable=consider-using-from-import
# pylint: disable=line-too-long
# pylint: disable=super-with-arguments
import os
import pickle

import joblib
import mlflow
import pandas as pd
import torch
import torch.nn as nn
from mlflow.tracking import MlflowClient

from data_processing import prepare_X_y
from model_registry import register_model, transition_model
from plotting import plot_learning_curve


class LSTMModel(nn.Module):
    """
    input_size : number of features in input at each time step
    hidden_size : Number of LSTM units
    num_layers : number of LSTM layers
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()  # initializes the parent class nn.Module
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):  # defines forward pass of the neural network
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out


def define_model(device, input_size, num_layers, hidden_size, output_size):
    # Define the model, loss function, and optimizer
    model = LSTMModel(input_size, hidden_size, num_layers).to(device)
    print(model)
    return model


def create_dataloader(X_train, y_train, X_val, y_val, batch_size=16):
    # Create DataLoader for batch training
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Create DataLoader for batch training
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader


def train_model(
    data_loader, num_epochs, learning_rate, train_data_path=None, val_data_path=None
):
    # pylint: disable = not-callable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = 1
    num_layers = 2
    hidden_size = 64
    output_size = 1

    model = define_model(device, input_size, num_layers, hidden_size, output_size)
    num_epochs = 50

    loss_fn = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = data_loader["train_loader"]
    val_loader = data_loader["val_loader"]

    # Start MLflow run
    with mlflow.start_run() as run:

        mlflow.set_tag("developer", "Victor")

        # log data
        mlflow.log_param("train_data_path", train_data_path)
        mlflow.log_param("val_data_path", val_data_path)

        # log parameters
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("output_size", output_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", learning_rate)

        train_hist = []
        val_hist = []
        # Training loop
        for epoch in range(num_epochs):
            total_train_loss = 0.0

            # Training
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = loss_fn(predictions, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            # Calculate average training loss and accuracy
            average_train_loss = total_train_loss / len(train_loader)
            train_hist.append(average_train_loss)

            # log training loss
            mlflow.log_metric("avg_train_loss", average_train_loss, step=epoch)

            # Validation on test data
            model.eval()
            with torch.no_grad():
                total_val_loss = 0.0

                for batch_X_val, batch_y_val in val_loader:
                    batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(
                        device
                    )
                    predictions_val = model(batch_X_val)
                    val_loss = loss_fn(predictions_val, batch_y_val)

                    total_val_loss += val_loss.item()

                # Calculate average test loss and accuracy
                average_val_loss = total_val_loss / len(val_loader)
                val_hist.append(average_val_loss)

            # log validation loss
            mlflow.log_metric("avg_val_loss", average_val_loss, step=epoch)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}"
                )

        # Log model
        mlflow.pytorch.log_model(model, artifact_path="LSTM-model")

    # End MLflow run
    mlflow.end_run()

    return model, run, train_hist, val_hist


def save_model(model, model_dir: str, model_format: str = "pickle"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"fuel_price_lstm.{model_format}")

    if model_format == "pickle":
        with open(model_path, "wb") as f_out:
            pickle.dump((model), f_out)

    if model_format == "joblib":
        joblib.dump(model, model_path)

    print(f"Model saved locally at: {model_path}")


def mlflow_registry(client, run, model_name: str):

    register_model(run.info.run_id, model_name)
    print("Model registered in MLflow")

    latest_versions = client.get_latest_versions(name=model_name)
    version = latest_versions[-1].version
    stage = "Production"
    transition_model(client, model_name, version, stage)

    print(f"The model version {version} was transitioned to {stage}")


if __name__ == "__main__":

    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME = "fuel-price-experiment"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    train_data_path = "../data/2024_train_data.parquet"
    train_data = pd.read_parquet(train_data_path)
    X_train, y_train = prepare_X_y("training", train_data, sequence_length=48)

    val_data_path = "../data/2024_val_data.parquet"
    val_data = pd.read_parquet(val_data_path)
    X_val, y_val = prepare_X_y("validation", val_data, sequence_length=24)

    train_loader, val_loader = create_dataloader(
        X_train, y_train, X_val, y_val, batch_size=16
    )

    num_epochs = 50
    learning_rate = 1e-3
    data_loaders = {"train_loader": train_loader, "val_loader": val_loader}

    trained_model, run, train_hist, val_hist = train_model(
        data_loaders, num_epochs, learning_rate, train_loader, val_loader
    )
    print(f"Current MLflow run id: {run.info.run_id}")

    model_dir = "models"
    model_format = "pickle"
    save_model(trained_model, model_dir, model_format)

    model_name = "fuel-price-predictor"
    mlflow_registry(client, run, model_name)

    plot_learning_curve(num_epochs, train_hist, val_hist)
