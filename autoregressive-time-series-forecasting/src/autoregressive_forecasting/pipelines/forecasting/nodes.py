import logging

import mlflow
import pandas as pd
from autoregressive_forecasting.helpers.utils import calculate_forecasting_metrics
from autoregressive_forecasting.helpers.utils import load_model
from statsforecast import StatsForecast


logger = logging.getLogger(__name__)


def forecast_time_series(
    model_input: pd.DataFrame,
    model_dict: dict,
    fallback_model_dict: dict,
    fcst_options: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fits and forecasts time series input. Returns forecast and in-sample prediction dataframes.

    Args:
        model_input (pd.DataFrame): input dataframe
        model_dict (dict): forecasting model specification
        fallback_model_dict (dict): fallback model specification
        fcst_options (dict): forecasting options (frequency, horizon, n_jobs)

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: forecast and in-sample prediction dataframes
    """
    logger.info(
        f"Loading model {model_dict['name']} with parameters={model_dict['parameters']}"
    )
    model = load_model(model_dict["name"])(**model_dict["parameters"])

    logger.info(
        f"Loading model {fallback_model_dict['name']=} with parameters={fallback_model_dict['parameters']}"
    )
    fallback_model = load_model(fallback_model_dict["name"])(
        **fallback_model_dict["parameters"]
    )

    sf = StatsForecast(
        df=model_input,
        models=[model],
        freq=fcst_options["frequency"],
        fallback_model=fallback_model,
        n_jobs=fcst_options["n_jobs"],
    )
    logger.info(f"Forecasting with {fcst_options['horizon']=} steps ahead.")
    forecast = sf.forecast(h=fcst_options["horizon"], fitted=True)
    in_sample_predictions = sf.forecast_fitted_values()
    return forecast, in_sample_predictions


def log_training_artifacts(
    in_sample_predictions: pd.DataFrame, model_name: str
) -> None:
    """Logs training artifacts to MLflow.

    Args:
        in_sample_predictions (pd.DataFrame): in-sample predictions
        model_name (str): model name
    """
    logger.info("Logging in-sample metrics")
    mlflow.log_metric("y_true_sum", in_sample_predictions.loc[:, "y"].sum())
    mlflow.log_metric("y_pred_sum", in_sample_predictions.loc[:, model_name].sum())
    loss_metrics = calculate_forecasting_metrics(in_sample_predictions, model_name)
    mlflow.log_metrics(loss_metrics)
