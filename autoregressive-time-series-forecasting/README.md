# Autoregressive time-series forecasting

## Contents

- [Overview](#overview)
- [Data](#data)
- [Methods](#methods)
- [Pipelines](#pipelines)
  - [Data Preprocessing](#dataprep)
  - [Forecasting](#forecasting)
  - [Temporal Cross-Validation](#cross-validation)
  - [Forecasting with exogenous variables](#forecasting-exo)
  - [End-to-end pipelines](#end-to-end-pipelines)
- [How to run](#how-to-run)

## Overview <a name="overview"></a>

This QuickStart ML Blueprints use case is supposed to serve as a basic example of a typical time-series forecasting application. This particular showcase features forecasting the sales of 45 stores in different regions and is based on [Walmart Stores Sales](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/overview) data.

Time-series forecasting enables businesses to make informed decisions based on future predictions of key metrics. It helps anticipate future trends, stay ahead of competition and adapt to changing market conditions. Ultimately, it can help improve efficiency, reduce costs, and increase revenue.

This use case uses `statsforecast` library, which offers a collection of popular univariate time series forecasting models optimized for high performance and scalability.

## Data <a name="data"></a>

This blueprint is using the [Walmart Stores Sales](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/overview) from Kaggle. In order to use that data, you have to register on Kaggle website and accept the rules of competition.

The data consists of weekly sales from 45 Walmart stores from `2010-02-05` to `2012-10-26`.

## Methods <a name="methods"></a>

Why use autoregressive models?
- [M5](https://www.sciencedirect.com/science/article/pii/S0169207021001874) is a competition whose purpose is to learn from empirical evidence how to improve forecasting performance and advance theory and practice of forecasting. The results showed that:
  - only 7.5% teams managed to outperform the top performing benchmark (Exponential Smoothing with bottom-up reconciliation)
  - among the teams (415) that managed to outperform all the benchmarks:
	  - 5 obtained improvements greater than 20%
	  - 42 greater than 15%
	  - 106 greater than 10%
	  - 249 greater than 5%
- autoregressive models are off-the-shelf methods that don't require feature engineering, hyperparameter tuning that gradient boosting models require
- autoregressive models are **considerably faster** than their gradient boosting or neural network based forecasting solutions
- `statsforecast` is a time-series forecasting library that utilizes autoregressive approach:
  - with a simple and intuitive API
  - is optimized for high performance and provides out-of-the-box compatibility with `Spark`, `Dask`, and `Ray`
  - has many univariate methods already implemented

## Pipelines <a name="pipelines"></a>

`autoregressive-time-series-forecasting` use case currently consists of 4 main steps that include concrete pipelines:

- Data Preprocessing (`data_processing`)
- Forecasting (`forecasting`)
- Temporal Cross-Validation (`cross_validation`)
- Forecasting with exogenous variables (`forecasting_with_exo_vars`)

Pipelines are linked with each other with:

- datasets that are stored in project's `/data` directory,
- artifacts and metadata stored in `MLflow`,
- configuration files are stored in `/conf` directory

### Data Preprocessing <a name="dataprep"></a>

`kedro run -p data_processing`

The 'data_processing' pipeline preprocesses data and prepares a dataframe for model forecasting.

The pipeline takes in raw data and performs data cleaning and transformation steps to create a prepared dataframe.

The output dataframe contains three columns:
- 'unique_id': A unique identifier for the time series,
- 'ds': A time index column that can be either an integer index or a datestamp,
- 'y': A column representing the target variable or measurement we wish to forecast.

The 'data_processing' pipeline ensures that the resulting dataframe is ready to be used as input to a forecasting model.

### Forecasting <a name="forecasting"></a>

`kedro run -p forecasting`

The 'forecasting' pipeline forecasts time series input, and logs in sample metrics to MLflow.

The pipeline takes in model input, loads a model and fallback model from the `statsforecast` library,
forecasts `params:forecast_options.horizon` steps ahead, and logs in sample metrics to `MLflow`.
A fallback model is a secondary model that is designed to be used in case the primary model fails to make predictions.

The output dataframe (forecast) contains three columns:
- 'unique_id': A unique identifier for the time series,
- 'ds': A time index column that can be either an integer index or a datestamp,
- 'params:model.name': A column representing the point predictions for time series.

### Temporal Cross-Validation <a name="cross-validation"></a>

`kedro run -p cross_validation`

The 'cross_validation' pipeline performs temporal cross-validation for multiple models. You can specify which models you want to fit in `src/autoregressive_forecasting/pipelines/cross_validation/models.py` and `conf/base/parameters/cross_validation/yml`.

The pipeline takes in model input, efficiently fits multiple models, and calculates forecasting errors.
The error metrics are saved to MLflow for easy tracking and analysis.

### Forecasting with exogenous variables <a name="forecasting-exo"></a>

`kedro run -p forecasting_with_exo_vars`

The 'forecasting_with_exogenous_vars' pipeline demonstrates how to use the AutoARIMA model from the `StatsForecast` library to forecast with exogenous variables.

The pipeline:
- takes in model input,
- splits it into train and test dataframes,
- fits two models on the training set (AutoARIMA with exogenous variables; AutoARIMA without exogenous variables),
- forecasts test dataframe using these 2 models,
- saves metrics to MLflow.

The output dataframe (forecast_exogenous) contains 5 columns:
- 'unique_id': A unique identifier for the time series,
- 'ds': A time index column that can be either an integer index or a datestamp,
- 'AutoARIMA_exogenous': A column representing the point predictions with exogenous variables for time series,
- 'AutoARIMA': A column representing the point predictions without exogenous variables for time series,
- 'y': A column representing the true values for time series.

It is important to note that:
- to forecast values in the future, you will need to have future exogenous variable values available,
- exogenous variables must have numeric types,
- AutoARIMA model is the only model in the `StatsForecast` library that supports forecasting with exogenous variables.

### End-to-end pipelines <a name="forecasting-exo"></a>

```bash
kedro run -p end_to_end_forecasting
kedro run -p end_to_end_cv
kedro run -p end_to_end_forecasting_with_cv
```

The goal of end-to-end pipelines is to automate the entire process and reduce the amount of manual intervention required to take raw data and produce an actionable output.

End-to-end pipelines include:
- `end_to_end_forecasting`, which consists of `data_processing` and `forecasting`
- `end_to_end_cv`, which consists of `data_processing` and `cross_validation`
- `end_to_end_forecasting_with_cv`, which consists of `data_processing`, `forecasting` and `cross_validation`

## How to run <a name="howtorun"></a>

To run this example as is (without changing any configuration), you need to:

1. Create the working environment according to [instructions given in the main QuickStart ML documentation](https://github.com/getindata/quickstart-ml-blueprints)

2. Download [data](#data) and put it inside `/data/01_raw` directory

3. Run end-to-end pipelines:
```bash
# run end-to-end forecasting pipeline
kedro run -p end_to_end_forecasting

# run end-to-end temporal cross-validation pipeline
kedro run -p end_to_end_cv

# run end-to-end forecasting and temporal cross-validation pipeline
kedro run -p end_to_end_forecasting_with_cv
```

Miscellaneous steps:
- Run tests from within `autoregressive-time-series-forecasting` project folder to check if everything is properly set up:
```bash
pytest
```

- Run MLflow and optionally Kedro-Viz:
```bash
kedro mflow ui
kedro viz
```

You can also run MLflow or Kedro-Viz on a selected port. For Kedro-Viz, you can also visualize only selected pipelines and set up `autoreload` option that refreshes visualizations each time any changes to pipelines are made:

```bash
kedro mlflow ui --port 5001
kedro viz --autoreload --port 4142 --pipeline end_to_end_forecasting
```

If you are developing inside a Dev Container, even in the cloud, you are still able to access MLflow and Kedro-Viz services through your local browser thanks to VSCode's automatic port forwarding feature.