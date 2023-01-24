"""
This is a boilerplate pipeline 'explanation'
generated using Kedro 0.18.4
"""
import logging
from typing import Any, Dict, List, Tuple, Union
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure

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


def calculate_shap(abt_sample: pd.DataFrame, model: Any) -> List[np.ndarray]:
    """Create Explainer and calculate SHAP values for the ABT sample

    Args:
        abt_sample (pd.DataFrame): ABT sample
        model (Any): any fitted model with `predict_proba` method

    Returns:
        List[np.ndarray]: A set of calculated SHAP values
    """
    logger.info(
        f"Calculating SHAP values on a sample of {abt_sample.shape[0]} observations..."
    )

    _, num_cols, cat_cols, _ = extract_column_names(abt_sample)
    features_sample = abt_sample[num_cols + cat_cols]

    explainer = shap.KernelExplainer(model.predict_proba, features_sample)
    shap_values = explainer.shap_values(features_sample)

    return shap_values


def create_explanations(
    shap_values: List[np.ndarray],
    abt_sample: pd.DataFrame,
    model: Any,
    pdp_top_n: int = 5,
) -> Tuple[Any]:
    """Create a set of explanations for given model and data sample.

    Args:
        shap_values (List[np.ndarray]): previously calculated SHAP values
        abt_sample (pd.DataFrame): ABT sample
        model (Any): any fitted model with `predict_proba` method
        pdp_top_n (int): number of top N most important features to show on partial dependence plots. Defaults to 5.

    Returns:
        Tuple[Any]: a set of different explanations to be logged
    """
    logger.info("Creating and logging model explanations...")

    _, num_cols, cat_cols, _ = extract_column_names(abt_sample)
    features_sample = abt_sample[num_cols + cat_cols]

    shap_summary_plot = _create_shap_summary_plot(shap_values, features_sample)
    feature_importance = _calculate_feature_importance(
        shap_values, features_sample, output_form="dict"
    )
    partial_dependence_plots = _create_partial_dependence_plots(
        shap_values, features_sample, model, pdp_top_n
    )

    return shap_summary_plot, feature_importance, partial_dependence_plots


def _create_shap_summary_plot(
    shap_values: List[np.ndarray], features_sample: pd.DataFrame
) -> Figure:
    """Create SHAP summary plot.

    Args:
        shap_values (List[np.ndarray]): SHAP values
        features_sample (pd.DataFrame): features (num_cols and cat_cols) extracted from ABT sample

    Returns:
        Figure: SHAP summary plot
    """
    shap.summary_plot(
        shap_values, features=features_sample, plot_size=(10, 10), show=False
    )
    shap_summary_plot = plt.gcf()

    return shap_summary_plot


def _calculate_feature_importance(
    shap_values: List[np.ndarray],
    features_sample: pd.DataFrame,
    output_form: str = "dict",
) -> Union[dict, pd.DataFrame]:
    """Calculate percentage of mean SHAP value-based feature importance.

    Args:
        shap_values (List[np.ndarray]): SHAP values
        features_sample (pd.DataFrame): features (num_cols and cat_cols) extracted from ABT sample
        output_form (str, optional): One of 'dict', 'data_frame'. Defaults to "dict".

    Returns:
        Union[dict, pd.DataFrame]: percentage of mean SHAP value-based feature importance
    """
    allowed_output = ["dict", "data_frame"]
    assert (
        output_form in allowed_output
    ), f"Parameter output_form should be one of {allowed_output}"

    vals = np.abs(shap_values).mean(0)
    sum_vals = sum(vals)
    sum_vals_norm = sum_vals / sum(sum_vals)

    feature_importance_df = pd.DataFrame(
        list(zip(features_sample.columns, sum_vals_norm)),
        columns=["feature", "importance"],
    )
    feature_importance_df.sort_values(by=["importance"], ascending=False, inplace=True)

    if output_form == "dict":
        feature_importance_dict = {
            k: v
            for k, v in zip(
                feature_importance_df["feature"], feature_importance_df["importance"]
            )
        }

        return feature_importance_dict
    else:
        return feature_importance_df


def _create_partial_dependence_plots(
    shap_values: List[np.ndarray],
    features_sample: pd.DataFrame,
    model: Any,
    pdp_top_n: int = 5,
) -> Dict[str, Figure]:
    """Create partial dependence plots for a set of most important features.

    Args:
        shap_values (List[np.ndarray]): SHAP values
        features_sample (pd.DataFrame): features (num_cols and cat_cols) extracted from ABT sample
        model (Any): any fitted model with `predict_proba` method
        pdp_top_n (int, optional): any fitted model with `predict_proba` method. Defaults to 5.

    Returns:
        Dict[str, Figure]: a dictionary of partial dependence plots
    """
    feature_importance = _calculate_feature_importance(
        shap_values, features_sample, output_form="data_frame"
    )

    top_n_feaures = feature_importance.index[:pdp_top_n].to_list()
    partial_dependence_plots = dict()
    for idx in top_n_feaures:
        shap.plots.partial_dependence(
            idx,
            lambda x: model.predict_proba(x)[:, 1],
            features_sample,
            model_expected_value=True,
            feature_expected_value=True,
            show=False,
        )
        feature_name = feature_importance["feature"][idx]
        partial_dependence_plots[f"{feature_name}.png"] = plt.gcf()

    return partial_dependence_plots
