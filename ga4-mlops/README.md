# Propensity model on Google Analytics 4 data with classification algorithms

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
- [GetInData ML Framework: Google Analytics 4 Use Case](#getindata-ml-framework-google-analytics-4-use-case)
  - [Data](#data)
  - [Pipelines:](#pipelines)
  - [Creating a new use case using VSCode devcontainers (recommended) ](#creating-a-new-use-case-using-vscode-devcontainers-recommended-)
  - [Creating a new use case manually ](#creating-a-new-use-case-manually-)

## Data

[Google Analytics 4 Dataset](https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset)

[Data schema](https://support.google.com/analytics/answer/7029846?hl=en)

Data is retrieved directly from BigQuery public data by running parametrized SQL queries within `data_preprocessing_train` and `data_preprocessing_predict` , there is no need to create any local samples by hand.

## Pipelines:

Currently there are 2 end to end  pipelines implemented and tested locally:
- `end_to_end_training` - for batch training. It consists of 3 consecutive sub-pipelines: `data_preprocessing_train_valid_test`, `feature_engineering_train_valid_test`, `training`
- `end_to_end_prediction` - for batch predictions. It consists of: `data_preprocessing_predict`, `feature_engineering_predict`, `prediction`

There is also one additional pipeline `explanation_{subset}` in three variations for different subsets: `train`, `valid` and `test` that applies some global XAI techniques and logs results to MLflow.
