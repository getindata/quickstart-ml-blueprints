build:
	@ echo "building docker image"
	kedro docker build --docker-args=--platform=linux/amd64

push-tag:
	@ echo "tagging & pushing docker image"
	docker tag gid-ml-framework:latest europe-west4-docker.pkg.dev/gid-ml-framework/gid-ml-framework-hm-data/gid-ml-framework
	docker push europe-west4-docker.pkg.dev/gid-ml-framework/gid-ml-framework-hm-data/gid-ml-framework

push:
	@ echo "pushing docker image to gcp"
	docker push europe-west4-docker.pkg.dev/gid-ml-framework/gid-ml-framework-hm-data/gid-ml-framework

tag:
	@ echo "tagging docker image"
	docker tag gid-ml-framework:latest europe-west4-docker.pkg.dev/gid-ml-framework/gid-ml-framework-hm-data/gid-ml-framework

generate_token:
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
