# autoregressive-forecasting

This is your new Kedro project configured according to GID ML Framework principles. Modify this README as you develop your project, for now you will find here some basic info that you need to get started. For more detailed assistance please refer to the [Kedro documenation](https://kedro.readthedocs.io/en/stable/index.html) and [GID ML Framework documentation](https://github.com/getindata/gid-ml-framework).

Additionally to a blank Kedro template it features technological stack used in GID ML Framework, such as:
  - [Poetry](https://python-poetry.org/)
  - [pre-commit](https://pre-commit.com/) hooks
  - [Dockerfile](https://docs.docker.com/engine/reference/builder/) setup
  - [VSCode Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) for ease of development
  - [MLFlow integration](https://kedro-mlflow.readthedocs.io/en/stable/)
  - [GCP VertexAI Kedro integration](https://github.com/getindata/kedro-vertexai) with integration to other platforms to be added

 Apart from that, there are no pre-implemented nodes or pipelines here. For blueprints showing different machine learning use cases, please go to the main [GID ML Framework](https://github.com/getindata/gid-ml-framework) repo and feel free to take as much as you need from our examples.

# Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://kedro.readthedocs.io/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

# Setting up the project

Below there are short instructions on how to get the environment for your new project up and running. Detailed version with some remarks and specific cases described are available in [GID ML Framework documentation - TO BE UPDATED](https://gitlab.com/getindata/aa-labs/coe/gid-ml-framework/-/blob/docs-release/README.md).

## Local Setup using VSCode devcontainers (recommended)
This approach facilitates use of [VSCode devcontainers](https://code.visualstudio.com/docs/devcontainers/containers). It is the easiest way to set up the development environment. 

Prerequisites:
* [VSCode](https://code.visualstudio.com/) with [Remote development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) extension
* [Docker](https://www.docker.com/) with `/workspaces` entry in `Docker Desktop > Preferences > Resources > File Sharing`

Setting up:
1. Clone this repository and [open it in a container](https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container).
2. You're good to go!

## Local Manual Setup

The project is using pyenv Python version management. It lets you easily install and switch between multiple versions of Python. To install pyenv, follow [these steps](https://github.com/pyenv/pyenv#installation=) for your operating system.

To install a specific Python version use this command:
```bash
pyenv install 3.8.16
pyenv shell 3.8.16
```

### Virtual environment

It is recommended to create a virtual environment in your project:
```
python -m venv venv
source ./venv/bin/activate
```

### Installing dependencies with Poetry

To install libraries declared in the pyproject.toml you need to have `Poetry` installed. Install it from [here](https://python-poetry.org/docs/#installing-with-the-official-installer) and then run this command:
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
# How to run Kedro

You can run your Kedro project with:

```bash
kedro run
```

To run a specific pipeline:
```bash
kedro run -p "<PIPELINE_NAME>"
```

# Kedro plugins
### [Kedro-Viz](https://github.com/kedro-org/kedro-viz)
- visualizes Kedro pipelines in an informative way
- to run, `kedro viz --autoreload` inside project's directory
- this will run a server on `http://127.0.0.1:4141`


### [kedro-mlflow](https://github.com/Galileo-Galilei/kedro-mlflow)
- lightweight integration of `MLflow` inside `Kedro` projects
- configuration can be specified inside `conf/<ENV>/mlflow.yml` file
- by default, experiments are saved inside `mlruns` local directory
- to see all the local experiments, run `kedro mlflow ui`
