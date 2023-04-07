import logging
from typing import Callable

import numpy as np
import pandas as pd
from autoregressive_forecasting.helpers.utils import log_dataframe_as_html_artifact
from datasetsforecast.losses import mae
from datasetsforecast.losses import mape
from datasetsforecast.losses import rmse
from statsforecast import StatsForecast

from .models import ForecastingModels


logger = logging.getLogger(__name__)


def cross_validate_forecasting_models(
    model_input: pd.DataFrame, cv_options: dict, model_params: dict
) -> pd.DataFrame:
    """Fits a list of forecasting models through multiple training windows. Returns in-sample predictions.

    Args:
        model_input (pd.DataFrame): input dataframe
        cv_options (dict): cross validation params
        model_params (dict): model params

    Returns:
        pd.DataFrame: dataframe with forecasts (unique_id, ds, y, model_1, model_2, ...)
    """
    forecasting_models = ForecastingModels(
        **model_params, test_run=cv_options["test_run"]
    )
    models_list = forecasting_models.get_models_list()
    fallback_model = forecasting_models.load_fallback_model()
    h, step_size, n_windows, freq, n_jobs, verbose = map(
        cv_options.get, ("h", "step_size", "n_windows", "freq", "n_jobs", "verbose")
    )

    logger.info(
        f"""Running cross-validation with following params:
        {h=}, {step_size=}, {n_windows=}, {freq=}, {n_jobs=}, {verbose=}"""
    )
    sf = StatsForecast(
        models=models_list,
        freq=freq,
        fallback_model=fallback_model,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    logger.info(
        f"Running cross-validation with following models: {forecasting_models.get_model_names_list()}"
    )
    cross_validation_df = sf.cross_validation(
        df=model_input,
        h=h,
        step_size=step_size,
        n_windows=n_windows,
    )
    return cross_validation_df


def _calculate_cross_validation_loss(
    cross_validation_df: pd.DataFrame,
    loss_metric: Callable[[np.ndarray, np.ndarray], float],
) -> pd.DataFrame:
    """Calculates loss for each product/forecasting model over period/cutoff.
    Loss metric is averaged for each article/method combination.

    Args:
        cross_validation_df (pd.DataFrame): dataframe with forecasts (unique_id, ds, y, model_1, model_2, ...)
        loss_metric (Callable[[np.ndarray, np.ndarray], float]): loss function that:
            - takes in ground truth values and predictions
            - returns loss

    Returns:
        pd.DataFrame: dataframe with error for each product, and forecasting model
    """
    models = [
        model
        for model in cross_validation_df.columns
        if model not in ["unique_id", "ds", "y", "cutoff"]
    ]
    evals = []
    for model in models:
        eval_ = (
            cross_validation_df.groupby(["unique_id"])
            .apply(lambda x: loss_metric(x.loc[:, "y"].values, x.loc[:, model].values))
            .to_frame()
        )  # Calculates loss for every unique_id, model
        eval_.columns = [model]
        evals.append(eval_)
    evals = pd.concat(evals, axis=1)
    evals = evals.groupby(["unique_id"]).mean(
        numeric_only=True
    )  # Averages the error metrics for every combination of model and unique_id
    return evals


def _aggregate_error_metrics(cross_validation_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates multiple error metrics from evaluation dataframe.

    Args:
        cross_validation_df (pd.DataFrame): dataframe with forecasts (unique_id, ds, y, model_1, model_2, ...)

    Returns:
        pd.DataFrame: dataframe with multiple error metrics for each forecasting method
            with columns: ("cutoff", "models", "mae", "mape", "rmse")
    """
    metrics = [mae, mape, rmse]
    evals = []
    for metric in metrics:
        evals.append(
            _calculate_cross_validation_loss(cross_validation_df, metric)
            .mean()
            .to_frame(name=metric.__name__)
        )
    evals = pd.concat(evals, axis=1)
    evals.index.name = "models"
    return evals


def calculate_cross_validation_errors(
    cross_validation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluates cross validation forecasts for each cutoff.

    Args:
        cross_validation_df (pd.DataFrame): dataframe with forecasts (unique_id, ds, y, model_1, model_2, ...)

    Returns:
        pd.DataFrame: aggregated error metrics for each cutoff
    """
    errors_df = cross_validation_df.groupby(["cutoff"]).apply(_aggregate_error_metrics)
    return errors_df


def log_cross_validation_errors(errors_df: pd.DataFrame) -> None:
    """Logs cross_validation metrics to MLflow.

    Args:
        errors_df (pd.DataFrame): aggregated error metrics for each cutoff
    """
    log_dataframe_as_html_artifact(errors_df, "cross_validation", "error_metrics.html")
