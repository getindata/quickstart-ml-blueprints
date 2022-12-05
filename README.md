# GetInData ML Framework

Example how to create a new project (use case) locally on MacOS:  
[Creating a new use case](#new-use-case)

## Overview

Machine learning framework for solving standard analytical problems. Includes data storage, exploratory data analysis, feature engineering, modeling, monitoring etc.

Take a look at the [Kedro documentation](https://kedro.readthedocs.io) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://kedro.readthedocs.io/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## Data
Easiest way to download data is by creating Kaggle account and downloading data from [here](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data). You can also download tabular data from [Google Drive](https://drive.google.com/drive/folders/1sdV7_nGqC-MvE4GXwC12btCGDIZplHa4?usp=sharing) (and sample of images).

## Python version management

The project is using pyenv Python version management. It lets you easily install and switch between multiple versions of Python. To install pyenv, follow [these steps](https://github.com/pyenv/pyenv#installation=) for your operating system. You may also use other alternatives like [conda](https://stackoverflow.com/questions/70851048/does-it-make-sense-to-use-conda-poetry).

To install a specific Python version use this command:
```
pyenv install <version> # 3.8.12
```

## How to setup Poetry

The project is using Poetry depenendency management. To install Poetry, follow [these steps](https://python-poetry.org/docs/#installation) for your operating system.

If you wish to include the virtual environment folder in the project, make a following change in Poetry's config:
```
poetry config virtualenvs.in-project true --local
poetry config virtualenvs.create true --local
poetry config virtualenvs.in-project true
```

To specify version of Python to use in the project:
```
poetry env use <version/full_path> # 3.8.12
```
You may also specify a full path to your Python version

To initialise a pre-populated directory do the following:
```
cd gid-ml-framework
poetry init
```

## How to install dependencies with Poetry

To add and install dependencies with:
```
poetry add <package_name>

# dev dependencies
poetry add -D <package_name>
```

If you have the libraries declared in the pyproject.toml file, you may install them with this command:
```
poetry install
```

The commands for working with virtual environment:
```
# activating venv
poetry shell

# deactivating venv
exit
```

## How to prepare GPU environment with Conda and Poetry

First install miniconda and Poetry for you OS version. Next create conda-lock file with requirements:
```
conda create -n temp -c conda-forge mamba conda-lock poetry='1.*' python='3.8.12' && conda clean -afy
conda activate temp
conda-lock -k explicit --conda mamba
poetry add --lock torch=1.12.1 torchaudio=0.12.1 torchvision=0.13.1 conda-lock
conda activate base
conda env remove -n temp
```

Create new conda environment and install poetry dependencies:

```
conda create --name gid_ml_framework --file conda-linux-64.lock && conda clean -afy
conda activate gid_ml_framework
poetry install
```

## How to setup pre-commit

To setup pre-commit hooks: 
```
pre-commit install
```

## How to run Kedro

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, look at the `.coveragerc` file.


## Project dependencies

To generate or update the dependency requirements for your project:

```
kedro build-reqs
```

This will copy the contents of `src/requirements.txt` into a new file `src/requirements.in` which will be used as the source for [`pip-compile`](https://github.com/jazzband/pip-tools#example-usage-for-pip-compile). You can see the output of the resolution by opening `src/requirements.txt`.

After this, if you'd like to update your project requirements, please update `src/requirements.in` and re-run `kedro build-reqs`.

[Further information about project dependencies](https://kedro.readthedocs.io/en/stable/04_kedro_project_setup/01_dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, `catalog`, and `startup_error`.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to convert notebook cells to nodes in a Kedro project
You can move notebook code over into a Kedro project structure using a mixture of [cell tagging](https://jupyter-notebook.readthedocs.io/en/stable/changelog.html#cell-tags) and Kedro CLI commands.

By adding the `node` tag to a cell and running the command below, the cell's source code will be copied over to a Python file within `src/<package_name>/nodes/`:

```
kedro jupyter convert <filepath_to_my_notebook>
```
> *Note:* The name of the Python file matches the name of the original notebook.

Alternatively, you may want to transform all your notebooks in one go. Run the following command to convert all notebook files found in the project root directory and under any of its sub-folders:

```
kedro jupyter convert --all
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can run `kedro activate-nbstripout`. This will add a hook in `.git/config` which will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://kedro.readthedocs.io/en/stable/03_tutorial/05_package_a_project.html)

---

# Creating a new use case <a name="new-use-case"></a>

Locally on MacOS:

0. Clone ML Framework Repo and create a branch for new use case:
```
git clone git@gitlab.com:getindata/aa-labs/coe/gid-ml-framework.git
git checkout -b <branch-name>
git push origin <branch-name> 
```

1. Install homebrew:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
export PATH=/opt/homebrew/bin:$PATH
export PATH=/opt/homebrew/sbin:$PATH
```

2. Ensure that you have proper build environment for your OS:
```
brew install openssl readline sqlite3 xz zlib tcl-tk
```

3. Install pyenv:
```
brew install pyenv
```

4. Set up shell environment for pyenv (zsh example):
```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

5. Install kedro to create new kedro project (using conda):
```
conda create --name kedro python=3.8.12 -y
conda activate kedro
pip install kedro==0.18.3
```

6. Create new kedro project:
```
kedro new  # give <project_name>
conda deactivate
cd <project_name>
```

7. Install Python 3.8.12 and set as global version:
```
pyenv install 3.8.12
pyenv global 3.8.12
```

8. Install Poetry:
```
curl -sSL https://install.python-poetry.org | python3 -
```

9. Change settings to include virtual envs in the project:
```
poetry config virtualenvs.in-project true --local
poetry config virtualenvs.create true --local
poetry config virtualenvs.in-project true
```

10. Initialize Poetry in existing project (make sure you are inside kedro project):
```
poetry init
```

11. Specify Python version for Poetry:
```
# Change <username> or entire pyenv path
poetry env use /Users/<username>/.pyenv/versions/3.8.12/bin/python3
```

12. Install pre-commit:
```
brew install pre-commit
pre-commit install
```

13. Add kedro to Poetry env:
```
poetry add kedro==0.18.3
```

14. Add JupyterLab to Poetry env ad dev dependency:
```
poetry add -D jupyterlab
```

15. Activate Poetry env shell inside project directory to avoid using `poetry run`:
```
poetry shell
```

16. Run JupyterLab with Kedro:
```
kedro jupyter lab
```