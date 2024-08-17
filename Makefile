setup:
	pip install -r requirements.txt
	pre-commit install

qualilty_checks:
	isort .
	black .
	pylint --recursive=y.

mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db


training_pipeline:
	python src/workflow.py


predict:
	python src/predict.py


run_all: setup qualilty_checks mlflow training_pipeline predict
