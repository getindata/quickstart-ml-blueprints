"""
This is a boilerplate pipeline 'prediction'
generated using Kedro 0.18.4
"""
import logging
from typing import Any

import pandas as pd

from ..data_preparation_utils import extract_column_names

logger = logging.getLogger(__name__)


def predict(abt_predict: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Make predictions on a given data frame.

    Args:
        abt_predict (pd.DataFrame): data frame to predict on
        model (Any): any model with predict_proba method

    Returns:
        pd.DataFrame: data frame with predicted scores
    """
    logger.info("Applying model to get predictions...")

    info_cols, num_cols, cat_cols, _ = extract_column_names(abt_predict)

    scores = model.predict_proba(abt_predict[num_cols + cat_cols])[:, 1]

    predictions = abt_predict.loc[:, info_cols]
    predictions["y_score"] = scores

    return predictions
