"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.4
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_data(ga4_raw_data: pd.DataFrame):
    """_summary_

    Args:
        ga4_raw_data (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    logger.info(f"Table downloaded, shape: {ga4_raw_data.shape}")
    logger.info(ga4_raw_data.head(5))
