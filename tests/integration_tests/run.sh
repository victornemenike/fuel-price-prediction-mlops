#!/usr/bin/env bash

cd "$(dirname "$0")"

if ["${LOCAL_IMAGE_NAME}" == ""];then
    LOCAL_TAG='date +"%Y-%m-%d-%H-%M"'
    export LOCAL_IMAGE_NAME="fuel-price-prediction-service:${LOCAL_TAG}"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_TAG_NAME}"
    docker build -t ${LOCAL_IMAGE_NAME} ..
else:
    echo "no need to build ${LOCAL_IMAGE_NAME}"

docker run -d --rm -p 8080:8080 ${LOCAL_IMAGE_NAME}

sleep 5

python test_docker.py
