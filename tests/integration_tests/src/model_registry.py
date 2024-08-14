# pylint: disable=duplicate-code
import warnings
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore", category=FutureWarning)


def register_model(run_id, model_name):
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=model_name)


def transition_model(client, model_name, version, stage):
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=False,
    )

    date = datetime.today().date()
    client.update_model_version(
        name=model_name,
        version=version,
        description=f"The model version {version} was transitioned to {stage} on {date}",
    )


def main():
    model_name = "fuel-price-predictor"
    run_id = "f83b41f1ac2b450893bbcdaac0e5d157"

    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME = "fuel-price-experiment"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    register_model(run_id, model_name)

    latest_versions = client.get_latest_versions(name=model_name)
    version = latest_versions[-1].version
    stage = "Staging"
    transition_model(client, model_name, version, stage)


if __name__ == "__main__":
    main()
