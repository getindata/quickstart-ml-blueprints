# GetInData ML Framework: Google Analytics 4 Use Case

GetInData ML Framework use case covering the following areas:
- Usage of Google Analytics 4 data
- Standard classification models (with scores calibration)
- Batch scoring and online scoring
- Automated Exploratory Data Analysis
- Automated model explanations
- Data Imputation
- AutoML packages utilization
- Model ensembling
- Algorithm performance comparisons and reporting
- Automated retraining
- Automated monitoring
- Deployment in different environments (GCP, AWS, Azure, Kubeflow)

Example how to create a new project (use case) locally on MacOS:  
[Creating a new use case](#new-use-case)

## Data

[Google Analytics 4 Dataset](https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset)

[Data schema](https://support.google.com/analytics/answer/7029846?hl=en)

Data is retrieved directly from BigQuery public data by running parametrized SQL queries within `data_preprocessing_train` and `data_preprocessing_predict` , there is no need to create any local samples by hand.

## Pipelines:

Currently there are 2 end to end  pipelines implemented and tested locally:
- `end_to_end_training` - for batch training. It consists of 3 consecutive sub-pipelines: `data_preprocessing_train_valid_test`, `feature_engineering_train_valid_test`, `training`
- `end_to_end_prediction` - for batch predictions. It consists of: `data_preprocessing_predict`, `feature_engineering_predict`, `prediction`

There is also one additional pipeline `explanation_{subset}` in three variations for different subsets: `train`, `valid` and `test` that applies some global XAI techniques and logs results to MLflow.

## Creating a new use case <a name="new-use-case"></a>

Locally on MacOS with `zsh`:

0. Clone ML Framework Repo and create a branch for new use case:
```
git clone git@gitlab.com:getindata/aa-labs/coe/gid-ml-framework.git
git checkout -b <branch-name>
git push origin <branch-name> 
```

1. Install homebrew:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Add paths to PATH permanently by adding to `./zshrc`:

```
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
echo 'export PATH="/opt/homebrew/sbin:$PATH"' >> ~/.zshrc
```

2. Ensure that you have proper build environment for your OS:
```
brew install openssl readline sqlite3 xz zlib tcl-tk
```

3. To avoid them accidentally linking against a Pyenv-provided Python"
```
alias brew='env PATH="${PATH//$(pyenv root)\/shims:/}" brew'
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

Add path to PATH permanently by adding to `./zshrc`:

```
echo 'export "PATH=$HOME/.local/bin:$PATH"' >> ~/.zshrc
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

17. For VSCode:  

Change `terminal.integrated.cwd` setting to your use case main directory (`<name>`)


