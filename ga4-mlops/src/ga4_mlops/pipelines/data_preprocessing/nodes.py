"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.4
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_and_aggregate_data(ga4_data: pd.DataFrame) -> pd.DataFrame:
    """Select, aggregate and sample GA4 data using parametrized SQL query.

    Args:
        ga4_data (pd.DataFrame): dataset (train, valid, test or prediction)

    Returns:
        pd.DataFrame: initially prepared data
    """
    logger.info(f"Table downloaded, shape: {ga4_data.shape}")

    return ga4_data
