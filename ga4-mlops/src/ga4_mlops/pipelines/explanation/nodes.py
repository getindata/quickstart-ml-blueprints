"""
This is a boilerplate pipeline 'explanation'
generated using Kedro 0.18.4
"""
import logging
from warnings import filterwarnings

import mlflow
import pandas as pd
import shap
from xgboost import XGBClassifier

from ..data_preparation_utils import extract_column_names

logger = logging.getLogger(__name__)
filterwarnings(
    action="ignore", category=DeprecationWarning
)  # Otherwise shap generates DeprecationWarning: "`np.int` is a deprecated alias for the builtin `int`"


def sample_data(abt: pd.DataFrame, n_obs: int, seed: int) -> pd.DataFrame:
    """Sample model input data preserving target proportions.

    Args:
        abt (pd.DataFrame): input data frame
        n_obs (int): number of observations in a sample

    Returns:
        pd.DataFrame: data frame sample
    """
    original_n_obs = abt.shape[0]
    n_obs = max(1, min(n_obs, original_n_obs))
    logger.info(
        f"Sampling data for SHAP explanations. Original size: {original_n_obs}; Sample size: {n_obs}"
    )

    _, _, _, target_col = extract_column_names(abt)
    logger.info(f"Target name: {target_col}")

    original_proportions = abt[target_col].value_counts() / original_n_obs
    logger.info(f"Original target proportions:\n{original_proportions.to_string()}")

    frac = n_obs / original_n_obs
    abt_sample = (
        abt.groupby(target_col)
        .apply(lambda x: x.sample(frac=frac, random_state=seed))
        .reset_index(drop=True)
    )

    proportions = abt_sample[target_col].value_counts() / n_obs
    logger.info(f"Sample target proportions:\n{proportions.to_string()}")

    return abt_sample


def explain_model(abt_sample: pd.DataFrame, model: XGBClassifier) -> shap.Explainer:
    """_summary_

    Args:
        abt_sample (pd.DataFrame): _description_
        model (XGBClassifier): _description_

    Returns:
        shap.Explainer: _description_
    """
    logger.info("Building model explainer...")

    _, num_cols, cat_cols, _ = extract_column_names(abt_sample)

    mlflow.shap.log_explanation(model.predict_proba, abt_sample[num_cols + cat_cols])
