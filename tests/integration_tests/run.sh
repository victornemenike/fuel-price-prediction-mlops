#!/usr/bin/env bash

cd "$(dirname "$0")"

LOCAL_TAG='date +"%Y-%m-%d-%H-%M"'
LOCAL_IMAGE_NAME="fuel-price-prediction-service:${LOCAL_TAG}"


docker build -t ${LOCAL_IMAGE_NAME} ..

docker run -d --rm -p 8080:8080 ${LOCAL_IMAGE_NAME}

sleep 5

python test_docker.py
