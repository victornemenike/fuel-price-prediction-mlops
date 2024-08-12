FROM python:3.11-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["web-service/Pipfile", "web-service/Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["src/data_collection.py", "src/data_processing.py", "./"]
COPY ["src/utils.py", "src/plotting.py", "./"]
COPY ["src/train.py", "src/model_registry.py", "./"]
COPY ["web-service/predict.py", "/models/fuel_price_lstm.pickle", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]