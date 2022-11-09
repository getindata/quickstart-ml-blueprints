build:
	@ echo "building docker image"
	kedro docker build --docker-args '--platform=linux/amd64 --network=host --progress=plain'

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

run:
	@ echo "run pipeline"
	kedro vertexai -e cloud run-once -p santander_dgsr_kaggle_gr

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
