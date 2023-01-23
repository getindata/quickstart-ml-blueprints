"""
This is a boilerplate pipeline 'prediction'
generated using Kedro 0.18.4
"""
import logging

import numpy as np
import pandas as pd

from ..data_preparation_utils import extract_column_names

logger = logging.getLogger(__name__)


def create_predictions(
    abt_predict: pd.DataFrame,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    threshold: float = 0.5,
    classify_on_calibrated: bool = True,
) -> pd.DataFrame:
    """Create output table with predictions.

    Args:
        abt_predict (pd.DataFrame): prediction ABT
        raw_scores (np.ndarray): raw scores on prediction ABT with main model
        calibrated_scores (np.ndarray): calibrated scores on prediction ABT with calibrator model
        threshold (float, optional): classification threshold as a basis for assigning predicted class labels.
            Defaults to 0.5.
        classify_on_calibrated (bool, optional): whether to assign predicted class labels based on raw scores
            or calibrated scores. Defaults to True.

    Returns:
        pd.DataFrame: Output table with scores and predicted labels.
    """
    logger.info("Generating predictions...")

    info_cols, _, _, _ = extract_column_names(abt_predict)

    predictions = abt_predict.loc[:, info_cols]
    predictions["y_raw_score"] = raw_scores
    predictions["y_calibrated_score"] = calibrated_scores
    if classify_on_calibrated:
        predictions["y_predicted_label"] = np.where(
            predictions["y_calibrated_score"] > threshold, 1, 0
        )
    else:
        predictions["y_predicted_label"] = np.where(
            predictions["y_raw_score"] > threshold, 1, 0
        )

    return predictions
