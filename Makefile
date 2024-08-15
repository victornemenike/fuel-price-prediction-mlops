LOCAL_TAG:=$(shell `date +"%Y-%m-%d-%H-%M"`)
LOCAL_IMAGE_NAME:="fuel-price-prediction-service:${LOCAL_TAG}"

test:
	pytest tests/unit_tests

qualilty_checks:
	isort .
	black .
	pylint --recursive=y.

build: qualilty_checks test
	docker build -t {LOCAL_IMAGE_NAME} .

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} integration_test/run.sh

publish: build integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash scripts/publish.sh

setup:
	pip install -r requirements.txt
	pre-commit install
