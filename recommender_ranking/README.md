# Two-stage recommendation system on transactional data with ranking models

## Overview

The goal of `recommender_ranking` use case is to develop **product recommendations** based on data from previous transactions, as well as from customer and product metadata (including images and text descriptions). The data comes from [Kaggle competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview/description).

## Recommender Systems
The goal of a recommender models is to provide users (customers) with a set of items (products) they might be interested in.

The first challenge that comes with building recommender systems is the **scale**, you can have millions of users and thousands of items. Predicting the likelihood of purchase for every user-item pair is **infeasible** in most cases.

That's why modern, industry recommender systems consists of at least 2 stages:
- candidate generation (retrieval)
- candidate ranking (scoring)

![Miro diagram](https://user-images.githubusercontent.com/84588078/204501550-af6ec66d-16ff-4b77-890d-a9aefb3893cd.jpg)

### Candidate generation
In candidate generation stage, we select a reasonably relevant set of items. This stage is supposed to **narrow down** millions of items into hundreds of candidates for the **downstream ranking task**. It can consist of many methods (e.g. filtering approaches: previous purchases, items similar to previous purchases, popular/trending products, algorithmic approaches: matrix factorization, ANN, image/text embedding similarities, etc.) to deliver **diverse set of candidates**. These methods are supposed to be **fast** and more **computationally efficient** than the ranking phase.

### Ranking
Ranking stage is a **slower, but more precise** step to **score** and **rank** previously retrieved candidates. It is also a place, where we can **add features**, for example:
- customer features (age, sex, last purchase date, number of purchases, etc.)
- product features (product category, product color, number of sold units, etc.)
- customer-product features (candidate similarity to previously purchased products, etc.)

Ranking can be modeled as a **learning-to-rank** or **classification** task (e.g. `lambdarank` or `binary` objective for gradient boosting models. If deep learning is applied, the final layer is either a softmax over a catalog of items, or a sigmoid predicting the likelihood of user-item interaction).

## All Kedro Pipelines
Supplementary pipelines:
- exploratory_data_analysis
    - uses `pandas_profiling` to automatically generate EDA report. Saves artifacts in mlflow
    - it is possible to add manual EDA charts inside `manual_eda` function
- sample_data
    - transforms H&M data into smaller sample, which allows quicker experiment iteration
- image_resizer
    - resizes images to given resolution
- filter_latest_transactions
    - filters to latest transactions, because there is no need for over 2 years of transactions data
- train_val_split
    - splits transactions data into training and validation

Feature engineering:
- image_embeddings
    - trains an autoencoder model with `PyTorch Lightning` using articles' images, and saves the model to `MLflow`'s model registry
- image_embeddings_inference
    - loads the model from the model registry, and creates embeddings for each article image
- text_embeddings
    - using `SentenceTransformer` creates embeddings for articles' descriptions
- feature_engineering_automated
    - creates automated features using `featuretools`
- feature_egineering_manual
    - creates manual features using basic transformations including: average number of days between transactions, percentage of sales online/offline, days since last transaction for each customer, etc.
- feature_selection
    - after creating features, removes highly null, single value and correlated features
    - saves columns, which should be keeped to a pickle
- feature_selection_apply
    - selects columns from `feature_selection` pipeline

Candidate generation pipelines:
- candidate_generation
    - generates candidates for each customer, some example methods include: [most popular articles](https://gitlab.com/getindata/aa-labs/coe/gid-ml-framework/-/blob/main/recommender_ranking/src/gid_ml_framework/pipelines/candidate_generation/nodes.py#L16), [most popular articles by age](https://gitlab.com/getindata/aa-labs/coe/gid-ml-framework/-/blob/main/recommender_ranking/src/gid_ml_framework/pipelines/candidate_generation/nodes.py#L127), [previously bought articles](https://gitlab.com/getindata/aa-labs/coe/gid-ml-framework/-/blob/docs-restructured-repo/recommender_ranking/src/gid_ml_framework/pipelines/candidate_generation/nodes.py#L223), [articles similar to previously bought using image/text embeddings](https://gitlab.com/getindata/aa-labs/coe/gid-ml-framework/-/blob/docs-restructured-repo/recommender_ranking/src/gid_ml_framework/pipelines/candidate_generation/nodes.py#L357)
- candidate_generation_validation
    - used together with `train_val_split` pipeline, allows to run multiple candidate_generation experiments with the goal of maximizing recall
    - results are logged to `mlflow`

Candidates feature engineering:
- candidates_feature_engineering
    - creates candidates features like: jaccard similarity, cosine image/text embeddings similarity between articles previously bought by customer and candidate articles
    - it is a separate pipeline, because calculating features for all possible combinations would be computationally expensive
- merge_candidate_features
    - merges previously created automated and manual features to candidate features

Ranking:
- ranking
    - trains a single `LightGBM` model
- ranking_optuna
    - runs hyperparameter search using `optuna` package
- recommendation_generation
    - given single or multiple models, generates recommendations for each customer in a Kaggle compliant format 

*All available pipelines are registered in `src/gid_ml_framework/pipeline_registry.py` file.*

## How to Run
[Kedro](https://github.com/kedro-org/kedro) framework was used to create maintainable and modular data science code. Take a look at the Kedro [documentation](https://kedro.readthedocs.io) to get started.

### H&M end-to-end recommendations
1. Train and generate image & text embeddings (`train_and_generate_embeddings`)
- train autoencoder model on image data
- generate image embeddings for each article
- generate text embeddings for each article

2. Maximize recall for candidates generation (`candidate_generation_training`)
- generate candidates using multiple, diverse set of methods
- evaluate candidates
- log recall to `MLflow`
- you can adjust parameters (`candidate_generation.yml`) to maximize recall

3. Maximize Mean Average Precision (MAP) for recommender model (`end_to_end_ranking_training`)
- generate candidates using multiple, diverse set of methods
- create features for candidates
- train a ranking model to score candidates
- log MAP to `MLflow`
- you can adjust parameters (`ranking.yml`) to maximize MAP

4. Generate final recommendations (`end_to_end_ranking_inference`)
- recalculate candidates for the most recent data
- recalculate features for the most recent data
- generate recommendations based on previously trained ranking model
- save recommendations in a `recommendations.csv` file

### Running end-to-end **sample** recommendations
- local
```bash
kedro run -e test_sample_local -p "train_and_generate_embeddings"
kedro run -e test_sample_local -p "candidate_generation_training"
kedro run -e test_sample_local -p "end_to_end_ranking_training"
kedro run -e test_sample_local -p "end_to_end_ranking_inference"
```

- cloud
```bash
kedro vertexai -e test_sample_cloud run-once -p "train_and_generate_embeddings"
kedro vertexai -e test_sample_cloud run-once -p "candidate_generation_training"
kedro vertexai -e test_sample_cloud run-once -p "end_to_end_ranking_training"
kedro vertexai -e test_sample_cloud run-once -p "end_to_end_ranking_inference"
```

### Running tests
To run tests, run the following command:
```bash
pytest src/tests/
```

---

## Data
You can download data from multiple sources:
- Easiest way to download data is by creating Kaggle account and downloading data from [here](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
- [Google Cloud Storage - all data](https://console.cloud.google.com/storage/browser/gid-ml-framework-hm-data;tab=objects?project=gid-ml-framework&prefix=&forceOnObjectsSortingFiltering=false)

Sample data is located here:
- [Google Cloud Storage - sample data](https://console.cloud.google.com/storage/browser/gid-ml-framework-hm-data/sample_data;tab=objects?project=gid-ml-framework&pageState=%28%22StorageObjectListTable%22%3A%28%22f%22%3A%22%255B%255D%22%29%29&prefix=&forceOnObjectsSortingFiltering=false)

*Remember to rename the sample data files, so they match the ones in the data catalog.*

---

## Local Setup using VSCode devcontainers (recommended)
This approach facilitates use of [VSCode devcontainers](https://code.visualstudio.com/docs/devcontainers/containers). It is the easiest way to set up the development environment. 

Prerequisites:
* [VSCode](https://code.visualstudio.com/) with [Remote development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) extension
* [Docker](https://www.docker.com/) with `/workspaces` entry in `Docker Desktop > Preferences > Resources > File Sharing`

Setting up:
1. Clone this repository and [open it in a container](https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container).
2. To use kedro vertexai properly you need to set up gcloud:
    ```
    gcloud auth login --update-adc
    gcloud config set project gid-ml-framework
    gcloud auth configure-docker europe-west4-docker.pkg.dev
    ```
3. You're good to go!

## Local Manual Setup

### Python version management

The project is using pyenv Python version management. It lets you easily install and switch between multiple versions of Python. To install pyenv, follow [these steps](https://github.com/pyenv/pyenv#installation=) for your operating system. You may also use other alternatives like [conda](https://stackoverflow.com/questions/70851048/does-it-make-sense-to-use-conda-poetry).

To install a specific Python version use this command:
```bash
pyenv install <version> # 3.8.16
```

### How to install dependencies with Poetry

To install libraries declared in the pyproject.toml file with this command:
```bash
poetry install
```

To add and install dependencies with:
```bash
# dependencies
poetry add <package_name>

# dev dependencies
poetry add -D <package_name>
```

The commands for working with virtual environment:
```bash
# activating venv
poetry shell

# deactivating venv
exit
```

---
## Kedro

### Rules and guidelines

* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`


### How to run Kedro

You can run your Kedro project with:

```bash
kedro run
```
> by default, Kedro runs the `__default__` pipeline from `src/pipeline_registry.py`

To run a specific pipeline:
```bash
kedro run -p "<PIPELINE_NAME>"
```

To see all the possible flags/options, run `kedro run -h`

### How to work with the project interactively?

```bash
kedro jupyter notebook
kedro jupyter lab
kedro ipython
```
To work with notebooks, you can load `context`, `catalog` and `startup_error` variables by running following cells:
```python
%load_ext kedro.extras.extensions.ipython
%reload_kedro
```

After loading, you can access **datasets** and **parameters** with `context`:
```python
context.catalog.load('name_of_dataset_from_data_catalog')
```

To see all datasets, run:
```python
[data for data in catalog.list() if not data.startswith('params:')]
```

### Kedro project structure
- project's code is located inside `src/gid_ml_framework` directory
- Kedro uses the pipeline design pattern, each pipeline is located inside `src/gid_ml_framework/pipelines` directory
- each pipeline has optional YAML parameters, which you can specify inside `conf/<ENV>/parameters` directory
- to create a new pipeline and run, you must:
    - create new pipeline `kedro pipeline create <NEW_PIPELINE_NAME>`
    - add recently created pipeline to `src/pipeline_registry.py`
    - run new pipeline `kedro run -p <NEW_PIPELINE_NAME>`
- each pipeline consists of `nodes.py` and `pipeline.py` files
    - basically, you can think of nodes as of building blocks that can be combined in pipelines to build workflows
    - inside `nodes.py` there are Python functions to be executed within a particular pipeline
    - inside `pipeline.py` you can specify the inputs, outputs, functions and parameters of a pipeline
    - by default, Kedro uses `SequentialRunner`, so pipelines are executed sequentially. There must not be any circular dependencies
- you can specify dataset location inside `conf/<ENV>/catalog.yml` file
- you can keep your local data inside `data/` directory, by default all data files are in `.gitignore` so you can't commit them to your repository

### Custom Kedro objects
You can implement your own custom Kedro objects:
- datasets - to be used within `conf/<ENV>/catalog.yml`
    - implement your dataset class inside `src/gid_ml_framework/extras/datasets` directory
    - at the minimum, a valid Kedro dataset needs to subclass the base `AbstractDataSet` and provide an implementation for the following abstract methods: `_load`, `_save` and `_describe`
- hooks - can be used before/after specific execution points of project
    - examples of hooks: `before_node_run`, `after_node_run`, `on_node_error`
    - you can write custom hooks inside `src/gid_ml_framework/hooks.py` file
    - to register a hook, you have to import and add it to `HOOKS` tuple inside `src/gid_ml_framework/settings.py` file
- runners - execution mechanisms used to run pipelines
    - examples of runners: `SequentialRunner` (default), `ParallelRunner`

To see all available custom Kedro objects and examples, check out Kedro official documentation.

---
## Kedro plugins
### [Kedro-Viz](https://github.com/kedro-org/kedro-viz)
- visualizes Kedro pipelines in an informative way
- to run, `kedro viz --autoreload` inside project's directory
- this will run a server on `http://127.0.0.1:4141`


### [kedro-docker](https://github.com/kedro-org/kedro-plugins/tree/main/kedro-docker)
Before running, make sure you have:
- installed Docker
- Docker daemon up and running

In order to build an image, run:
```bash
kedro docker build
```
> If you have a MacBook with ARM processor, you must add `--docker-args=--platform=linux/amd64` option to run docker in the cloud.

Once built, you can run your project in a Docker environment:
```bash
kedro docker run
```
> This command automatically adds `--rm` flag, so the container will be automatically removed when it exits.

You can run your Docker image interactively:
```bash
kedro docker ipython
kedro docker jupyter notebook
kedro docker jupyter lab
```

`kedro-docker` allows to analyze the size efficiency of your project image by leveraging `dive`:
```bash
kedro docker dive
```

### [kedro-mlflow](https://github.com/Galileo-Galilei/kedro-mlflow)
- lightweight integration of `MLflow` inside `Kedro` projects
- configuration can be specified inside `conf/<ENV>/mlflow.yml` file
- by default, experiments are saved inside `mlruns` local directory
- to see all the local experiments, run `kedro mlflow ui`

### [kedro-vertexai](https://github.com/getindata/kedro-vertexai)
- supports running workflows on GCP Vertex AI Pipelines
- configuration can be specified inside `conf/<ENV>/vertexai.yml` file
- to start Vertex AI Pipeline, run `kedro vertexai -e <ENV> run-once -p <PIPELINE_NAME>`
