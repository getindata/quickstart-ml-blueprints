build:
	@ echo "building docker image"
	docker build -t recommender_gnn . --network=host --progress=plain

tag:
	@ echo "tagging docker image"
	docker tag recommender_gnn:latest europe-west4-docker.pkg.dev/gid-ml-framework/gid-ml-framework-data/recommender_gnn

push:
	@ echo "pushing docker image to gcp"
	docker push europe-west4-docker.pkg.dev/gid-ml-framework/gid-ml-framework-data/recommender_gnn:latest

tag-push:
	@ echo "tagging & pushing docker image"
	docker tag recommender_gnn:latest europe-west4-docker.pkg.dev/gid-ml-framework/gid-ml-framework-data/recommender_gnn
	docker push europe-west4-docker.pkg.dev/gid-ml-framework/gid-ml-framework-data/recommender_gnn:latest

run-vertex:
	@ echo "run pipeline"
	kedro vertexai -e cloud run-once -p santander_dgsr_kaggle_gr

run:
	@ echo "run docker image"
	docker run --network=host --gpus all europe-west4-docker.pkg.dev/gid-ml-framework/gid-ml-framework-data/recommender_gnn:latest

run-interactive:
	@ echo "run interactive bash session"
	docker run -it --network=host --gpus all europe-west4-docker.pkg.dev/gid-ml-framework/gid-ml-framework-data/recommender_gnn:latest /bin/bash

run-custom:
	@ echo "runs custom vertexai job"
	gcloud ai custom-jobs create \
  	--region=europe-west4 \
 	--display-name=$(arg) \
  	--config=config.yaml

generate-token:
	@ echo "generating identity token"
	gcloud auth print-identity-token \
    --impersonate-service-account="iap-accessor@gid-advanced-analytics-infra.iam.gserviceaccount.com" \
    --audiences="260364966542-721p48q045h4o2hmsmg5dl0gksm7udiu.apps.googleusercontent.com" \
    --include-email

jupyter:
	poetry run kedro jupyter notebook

mlflow:
	poetry run kedro mlflow ui

kedro-viz:
	poetry run kedro viz --autoreload
