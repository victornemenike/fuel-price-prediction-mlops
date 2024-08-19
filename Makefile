setup:
	pip install -r requirements.txt
	pre-commit install

qualilty_checks:
	isort .
	black .
	pylint monitoring src tests web-service

mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db


training_pipeline:
	python src/ml_workflow.py


predict:
	python src/predict.py
