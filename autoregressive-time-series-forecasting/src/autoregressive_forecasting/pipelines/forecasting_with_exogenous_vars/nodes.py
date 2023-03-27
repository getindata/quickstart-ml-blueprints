import logging

import mlflow
import pandas as pd
from autoregressive_forecasting.helpers.utils import calculate_forecasting_metrics
from autoregressive_forecasting.helpers.utils import filter_dataframe_by_date_cutoffs
from autoregressive_forecasting.helpers.utils import rename_columns
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA


logger = logging.getLogger(__name__)


def _sum_sales(df: pd.DataFrame, exo_columns: list[str]) -> pd.DataFrame:
    """Sums sales for each store and date.

    Args:
        df (pd.DataFrame): input dataframe
        exo_columns (list[str]): list containing exogenous columns

    Returns:
        pd.DataFrame: aggregated sales
    """
    agg_cols = exo_columns + ["unique_id", "ds"]
    store_sales = df.groupby(agg_cols)["y"].sum().reset_index()
    return store_sales


def _split_train_test(
    df: pd.DataFrame, no_test_periods: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits time-series into train and test datasets based on number of test periods (`no_test_periods`).

    Args:
        df (pd.DataFrame): dataframe with "ds" column (datestamp)
        no_test_periods (int): number of test periods, which will be in test dataset

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: respectively (train, test and test target) dataframes
    """
    test_periods = (
        df.loc[:, "ds"]
        .drop_duplicates()
        .sort_values(ascending=False)
        .head(no_test_periods)
        .unique()
    )
    df_train = df[~(df.loc[:, "ds"].isin(test_periods))]
    X_test = df[df.loc[:, "ds"].isin(test_periods)].drop(["y"], axis=1)
    Y_test = df[df.loc[:, "ds"].isin(test_periods)].loc[:, ["unique_id", "ds", "y"]]
    return df_train, X_test, Y_test


def preprocess_data(
    df: pd.DataFrame,
    mapper: dict[str, str],
    no_test_periods: int,
    test_run: bool,
    exo_columns: list[str],
    min_date_cutoff: str | None = None,
    max_date_cutoff: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepares data for training:
        - changes column names
        - splits dataframe into train, test and target test dataframes

    Args:
        df (pd.DataFrame): input dataframe
        mapper (dict[str, str]): mapper from raw dataframe column names to ["unique_id", "ds", "y"]
        no_test_periods (int): number of test periods, which will be in test dataset
        test_run (bool): boolean flag whether to run forecast on sample articles
        exo_columns (list[str]): list containing exogenous columns
        min_date_cutoff (str | None): minimum date to include in the filtered dataframe. Defaults to None
        max_date_cutoff (str | None): maximum date to include in the filtered dataframe. Defaults to None

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: respectively (train, test and test target) dataframes
    """
    df_renamed = rename_columns(df, mapper)
    df_filtered = filter_dataframe_by_date_cutoffs(
        df_renamed, min_date_cutoff, max_date_cutoff
    )
    df_sum = _sum_sales(df_filtered, exo_columns)
    if test_run:
        logger.info("Selecting random 10 'unique_id' for test run.")
        random_articles = (
            df_sum["unique_id"].drop_duplicates().sample(10, random_state=123)
        )
        df_sum = df_sum[df_sum["unique_id"].isin(random_articles.to_list())]
    df_train, X_test, Y_test = _split_train_test(df_sum, no_test_periods)
    Y_test = Y_test.set_index("unique_id")
    return df_train, X_test, Y_test


def forecast_with_exogenous(
    df_train: pd.DataFrame,
    X_test: pd.DataFrame,
    fcst_options: dict,
    fit_exogenous: bool,
) -> pd.DataFrame:
    """Fits and forecasts time series input. Returns forecast dataframe.
    If `fit_exogenous=True`, then model uses exogenous variables. Exogenous columns must have numeric types.

    Args:
        df_train (pd.DataFrame): train dataframe with ["unique_id", "ds", "y"] columns and optionally exogenous columns
        X_test (pd.DataFrame): test dataframe with ["unique", "ds"] columns and optionally future exogenous columns
        fcst_options (dict): forecasting options (frequency, horizon)
        fit_exogenous (bool): boolean flag whether to forecast with or without exogenous variables

    Returns:
        pd.DataFrame: forecast dataframe
    """
    models = [AutoARIMA(season_length=fcst_options["season_length"])]

    sf = StatsForecast(
        models=models, freq=fcst_options["frequency"], n_jobs=fcst_options["n_jobs"]
    )

    if fit_exogenous:
        logger.info("Fitting model with exogenous variables")
        forecast = sf.forecast(
            df=df_train, h=fcst_options["horizon"], X_df=X_test
        ).rename(mapper={"AutoARIMA": "AutoARIMA_exogenous"}, axis="columns")
    else:
        df_train = df_train.loc[:, ["unique_id", "ds", "y"]]
        logger.info("Fitting model without exogenous variables")
        forecast = sf.forecast(df=df_train, h=fcst_options["horizon"])
    return forecast


def merge_dataframes(
    forecast_exo: pd.DataFrame, forecast: pd.DataFrame, Y_test: pd.DataFrame
) -> pd.DataFrame:
    """Merges forecast with exogenous, forecast without exogenous, and target test dataframes.

    Args:
        forecast_exo (pd.DataFrame): forecast with exogenous variables dataframe
        forecast (pd.DataFrame): forecast without exogenous variables dataframe
        Y_test (pd.DataFrame): target test dataframe

    Returns:
        pd.DataFrame: merged dataframe with ["unique_id", "ds", "y", "model_with_exo", "model_without_exo"]
    """
    forecast_exo = forecast_exo.merge(
        forecast, on=["unique_id", "ds"], how="inner"
    ).merge(Y_test, on=["unique_id", "ds"], how="inner")
    return forecast_exo


def log_exogenous_metrics(forecast_exo: pd.DataFrame) -> None:
    """Logs metrics to MLflow.

    Args:
        forecast_exo (pd.DataFrame): merged dataframe with
            ["unique_id", "ds", "y", "model_with_exo", "model_without_exo"] columns
    """
    models = [
        col for col in forecast_exo.columns if col not in ["ds", "y", "unique_id"]
    ]
    for model in models:
        with mlflow.start_run(run_name=model, nested=True):
            loss_metrics = calculate_forecasting_metrics(forecast_exo, model)
            mlflow.log_metrics(loss_metrics)
