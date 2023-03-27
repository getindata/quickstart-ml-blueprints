jupyter:
	poetry run kedro jupyter notebook

mlflow:
	poetry run kedro mlflow ui

kedro-viz:
	poetry run kedro viz --autoreload

build:
	@ echo "building docker image"
	kedro docker build --docker-args=--platform=linux/amd64 --base-image=python:3.10.10-slim
