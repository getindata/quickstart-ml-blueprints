# GetInData ML Framework

GetInData Machine Learning Framework is a set of complete blueprints for solving typical machine learning problems. It leverages best-in-class open source technologies and materializes best practices for structuring and developing machine learning solutions.

By releasing this repo of reusable examples we aim to help data scientists and machine learning engineers to prototype their solutions faster using well-proven tooling and keep the highest quality and maintainability of the code.

## Contents

- [Overview](#overview)
- [Use cases](#usecases)
- [Technologies](#technologies)
- [How to start](#howtostart)
    - [Creating a new project](#howtostart-new)
    - [Running existing project locally](#howtostart-local)
        - [Recommended way using VSCode and Dev Containers](#howtostart-local-vsc)
        - [Remarks on some technologies/setups/operating systems](#howtostart-local-remarks)
        - [Alternative ways of manual environment creation](#howtostart-local-alt)
    - [Running existing project on GCP (VertexAI)](#howtostart-gcp)
    - [Running existing project on other full-scale environments](#howtostart-other)
- [Working with GID ML Framework](#wayofwork)

## Overview <a name="overview"></a>

A brief summary of why GID ML Framework was brought to life:
- to decrease time-to-market by enabling faster PoCs
- to share best practices for developing ML products
- to organize and standardize Way-of-Work for data scientists

To achieve the above goals, we do not aim to create any ML platform or any specific code package. On the other hand, we also do not want to only share theoretical insights about way-of-work or describe our project experience. Instead, we are creating a **library of solved machine learning use cases that implement ML code development best practices using modern open-source technology stack.** Some of the most important features tha our GID ML Framework provides are:
- Transferable environments (local/cloud)
- Production quality code from the start
- Well-organized configuration
- Careful dependency management
- Experiment tracking and model versioning
- Node/pipeline-based modular workflow that is fully reproducible and reusable
- Comprehensive documentation and test coverage
- Predefined, standard use cases (blueprints)

Apart from materializing best development practices and standardizing problem solving approach, the motivation for creating GID ML Framework results from observation, that many business problems that are solved using machine learning can be described as an interconnected collection of repeatable building blocks. If you pre-define and implement those building blocks on some real-life examples and do so in a well-structured modular way, those elements will become easily reusable in different, similar use cases that you may encounter. Reusing existing building blocks with a minimum modifications should make **prototyping of new solutions much more efficient, and also facilitate creating a well-structured, documented and tested production grade code from the very beginning of the project**.

![A generic modular scheme of a machine learning prototype solution fitting 99% of typical business use cases](./docs/img/generic_scheme.png)

## Use cases <a name="usecases"></a>

GID ML Framework is a set of complete **use cases** that involve:
- a definition of a business problem that is being solved
- a specific machine learning approach to solve this problem
- example datasets to work with

So far the following use cases have been implemented:
- propensity-to-buy classification model on Google Analytics 4 data with additional MLOps components ([ga4-mlops - TO BE UPDATED](https://gitlab.com/getindata/aa-labs/coe/gid-ml-framework/-/tree/main/ga4-mlops))
- retail recommender system on multimodal data (tabular, images, natural language descriptions) using ranking gradient boosting models ([recommender_ranking - TO BE UPDATED](https://gitlab.com/getindata/aa-labs/coe/gid-ml-framework/-/tree/main/recommender_ranking))
- e-commerce recommender system on sequential data using Graph Neural Networks ([recommender_gnn - TO BE UPDATED](https://gitlab.com/getindata/aa-labs/coe/gid-ml-framework/-/tree/main/recommender_gnn))

More use cases are either in works or in plans.

Existing use cases are implemented in modular, modifiable and extensible way. When creating a new ML solution, example building blocks from other, similar use cases can be used to various extent. In some cases, even small modifications to existing examples can be sufficient to obtain first working prototypes. For example, if the user is facing a problem of predicting churn and plans to approach it using classification algorithms, he can basically take [ga4-mlops - TO BE UPDATED](https://gitlab.com/getindata/aa-labs/coe/gid-ml-framework/-/tree/main/ga4-mlops) as a blueprint, change configuration files to use his datasets, modify data preprocessing code and run the rest of the pipeline as is. Since both the flavor of input data (tables with binary target per observation) and problem solving approach (binary classification) is similar, all automatic feature encoding, data imputation, modeling and model explanation steps should be applicable to this new problem - at least in a first iteration. More about the way of working with GID ML Framework and pre-implemented use cases can be foung in [this section](#wayofwork).

![From generic scheme to specific use cases](./docs/img/use_cases.png)

## Technologies <a name="technologies"></a>

To materialize ML development best practices as concrete working examples we use a modern technology stack. **Our main assumption is to stick to the state of the art, well-proven open source tooling.** We want the GID ML Framework to be applicable and adjustable to any MLOps architecture, so we avoid using commercial or proprietary software for essential functionalities.

![Technologies used so far](./docs/img/technologies.png)

Excerpt of the technologies used so far in existing examples:
- [Kedro](https://kedro.org/), which is the very core upon which the solutions are built. It introduces many essential features like appropriate project structure, modular node/pipeline architecture, well-organized configuration, customizable data catalog with connectors to many data sources, a wide variety of extensions and plugins that allow for integration with other tools and more.
- Visual Studio Code with [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) powered by Docker that enable creating encapsulated working environments with all necessary tooling that can be run and used in the same way no matter when they are deployed (access from local IDE to the working environment set up locally or in the cloud)
- [Pyenv](https://github.com/pyenv/pyenv) and [Poetry](https://python-poetry.org/) for Python dependency management
- GetInData Kedro plugins for running Kedro pipelines in different environments ([GCP](https://github.com/getindata/kedro-vertexai), [AWS](https://github.com/getindata/kedro-sagemaker), [Azure](https://github.com/getindata/kedro-azureml), [Kubeflow](https://github.com/getindata/kedro-kubeflow))
- [MLflow](https://mlflow.org/) as an experiment tracker and model repository
- A set of linters and code quality tools ([flake8](https://flake8.pycqa.org/en/latest/), [Black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/), [pre-commit](https://pre-commit.com/))
- [Pytest](https://docs.pytest.org/) framework for writing and executing unit/integration tests
- A collection of Python packages for automated exploratory analysis ([Pandas Profiling](https://ydata-profiling.ydata.ai/docs/master/index.html), [Featuretools](https://www.featuretools.com/)), modeling ([LightGBM](https://lightgbm.readthedocs.io/), [PyTorch](https://pytorch.org/) and more), hyperprameter tuning ([Optuna](https://optuna.org/)) and many more

## How to start <a name="howtostart"></a>

### Creating a new project <a name="howtostart-new"></a>

The best way to create a new project is to use GID ML Framework starter. The repository and instructions can be found [here - TO BE UPDATED](https://gitlab.com/getindata/aa-labs/coe/gid-ml-framework-starter/-/tree/main). After creating a new project, follow the guidelines for existing projects.

### Running existing project locally <a name="howtostart-local"></a>

GID ML Framework tries to address the challenge of building reproducible working environments that should be consistent, easy to establish and portable allowing data scientist to create small-scale prototypes locally and then seamlessly move their development to full-scale platforms, either in the cloud (VertexAI, Sagemaker, AzureML) or in on-premises setups (using Kubeflow). We use a combination of Pyenv, Poetry and Docker and leverage Visual Studio Code's Dev Containers to create a recommended development setup, but keeping in mind that there can be always some platform specific nuances (see for example a note from our experience [here](#howtostart-local-remarks)) we leave the freedom of adjusting way-of-work to the specific needs.

#### Recommended way using VSCode and Dev Containers <a name="howtostart-local-vsc"></a>

VSCode's [Remote Development](https://code.visualstudio.com/docs/remote/remote-overview) and [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) features brought together create quite unique opportunity to encapsulate your working environment inside a Docker container and connect to it from VSCode IDE no matter where the container is deployed. Following our approach, you can build a working environment just by opening your locally stored project folder inside a container, and later do exactly the same, but this time with the same container built not on your local machine but e.g. on a virtual machine in the cloud. The way you work doesn't change at all - you have your local IDE with all settings and favorite plugins, you just connect to a different backend. All technicalities like re-building the container if you update you working environment or port forwarding for in-project services like [MLflow](https://mlflow.org/) or [Kedro-Viz](https://kedro.readthedocs.io/en/stable/visualisation/kedro-viz_visualisation.html) is handled by VSCode.

The steps to run existing or newly created project are as follows:

1. Get prerequisites:
    - [VSCode](https://code.visualstudio.com/) with [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) extension
    - [Docker](https://www.docker.com/) with `/workspaces` entry in `Docker Desktop > Preferences > Resources > File Sharing`

2. Either create a new project using our [Kedro starter](https://gitlab.com/getindata/aa-labs/coe/gid-ml-framework-starter) or clone `gid-ml-framework` repository and open folder with selected use case in VSCode.

3. If Docker is running, VSCode should should ask to ["Reopen Folder in a Container"]((https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container)). You can also bring it up manually by clicking on the blue arrows icon in the bottom-left corner in VSCode.

4. After reopening the project folder in a container, the Dev Container will be built. It might take a few minutes at first attempt, later cache should be used if there is a need to rebuild a containers. As you work on the project, you can modify your environment configuration files (e.g. `poetry.toml`, `pyproject.toml`, `Dockerfile`, `devcontainer.json` etc.). You can do it either from inside or outside of the container - changes will be detected and VSCode will suggest to rebuild the container.
 
5. From now on, you can develop inside the container and modify and add files the usual way as you would work 100% locally. However, since the container is an isolated environment, you will need to configure git (e.g. SSH keys) or cloud connection (for projects transferrable to cloud) form inside of the container.

#### Remarks on some technologies/setups/operating systems <a name="howtostart-local-remarks"></a>

- corporate environments (user-managed notebooks, SSH, downloading data locally)
- ARM64
- Windows

#### Alternative ways of manual environment creation <a name="howtostart-local-alt"></a>

### Running existing project on GCP (VertexAI) <a name="howtostart-gcp"></a>

### Running existing project on other full-scale environments <a name="howtostart-other"></a>

Work on testing other full-scale environments (AWS, Azure, Kubeflow) is in progress.

## Working with GID ML Framework <a name="wayofwork"></a>
