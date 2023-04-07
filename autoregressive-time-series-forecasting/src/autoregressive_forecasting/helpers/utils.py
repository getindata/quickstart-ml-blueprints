import logging
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

import mlflow
import pandas as pd
from datasetsforecast.losses import mae
from datasetsforecast.losses import mape
from datasetsforecast.losses import rmse
from statsforecast.models import _TS


logger = logging.getLogger(__name__)


def log_dataframe_as_html_artifact(
    df: pd.DataFrame,
    artifact_path: str,
    file_name: str,
) -> None:
    """Creates temporary directory, and logs dataframe as html artifacts to MLflow.

    Args:
        df (pd.DataFrame): dataframe to log
        artifact_path (str): artifact path at which to save artifact
        file_name (str): saved file name
    """
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / file_name
        df.to_html(path, float_format="{:10.2f}".format)
        mlflow.log_artifact(local_path=path, artifact_path=artifact_path)


def get_all_available_models() -> list[str]:
    """Lists all available models from statsforecast.models module.

    Returns:
        list[str]: all available models
    """
    all_models = getattr(import_module("statsforecast.models"), "__all__")
    return all_models


def load_model(model_name: str) -> _TS:
    """Loads model from statsforecast.models module.

    Args:
        model_name (str): model name

    Returns:
        _TS: model object
    """
    try:
        model = getattr(import_module("statsforecast.models"), model_name)
    except AttributeError:
        logger.error(f"Couldn't load {model_name=}\n")
        logger.error(f"All available models include: {get_all_available_models()}\n")
        raise
    return model


def calculate_forecasting_metrics(
    predictions: pd.DataFrame, model_name: str
) -> dict[str, float]:
    """Calculates forecasting regression metrics (MSE, MAE, ...)

    Args:
        predictions (pd.DataFrame): predictions
        model_name (str): model name

    Returns:
        dict[str, float]: loss metrics
    """
    metrics = [mae, rmse, mape]
    loss_metrics = dict()
    y_true = predictions.loc[:, "y"]
    y_pred = predictions.loc[:, model_name]
    for metric in metrics:
        loss_metrics[metric.__name__] = metric(y_true, y_pred)
    return loss_metrics


def rename_columns(df: pd.DataFrame, mapper: dict[str, str]) -> pd.DataFrame:
    """Renames the columns given the dictionary mapper.

    Args:
        df (pd.DataFrame): input dataframe
        mapper (dict[str, str]): mapper from raw dataframe column names to the column names we want

    Returns:
        pd.DataFrame: dataframe with changed column names
    """
    df_renamed = df.rename(mapper=mapper, axis="columns")
    return df_renamed


def filter_dataframe_by_date_cutoffs(
    df: pd.DataFrame, min_date_cutoff: str | None, max_date_cutoff: str | None
) -> pd.DataFrame:
    """Filter a pandas DataFrame by a minimum and maximum date cutoff.

    Args:
        df (pd.DataFrame): dataframe with daily sales
        min_date_cutoff (str | None): minimum date to include in the filtered dataframe
        max_date_cutoff (str| None): maximum date to include in the filtered dataframe

    Returns:
        pd.DataFrame: dataframe containing only rows where the date falls within the specified range
    """
    if not (min_date_cutoff or max_date_cutoff):
        logger.info("Skipping filtering")
        return df
    logger.info(
        dedent(
            f"""\
        Min available date = {df['ds'].min().strftime('%Y-%m-%d')}
        Max available date = {df['ds'].max().strftime('%Y-%m-%d')}
        Min date cutoff = {min_date_cutoff}
        Max date cutoff = {max_date_cutoff}"""
        )
    )
    logger.info(f"Number of rows before filtering {len(df)}")
    if min_date_cutoff:
        df = df.loc[df["ds"] >= min_date_cutoff]
    if max_date_cutoff:
        df = df.loc[df["ds"] <= max_date_cutoff]
    logger.info(f"Number of rows after filtering {len(df)}")
    if df.empty:
        raise ValueError("Filtered dataframe is empty, check date cutoff parameters")
    return df
