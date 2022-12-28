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


def concatenate_train_valid_test(
    df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame
) -> pd.DataFrame:
    """

    Args:
        df_train (pd.DataFrame): training subset
        df_valid (pd.DataFrame): validation subset
        df_test (pd.DataFrame): test subset

    Returns:
        pd.DataFrame: combined subsets with subset labels
    """
    logger.info("Combining and labeling train/valid test subsets...")

    df_train["subset"] = "train"
    df_valid["subset"] = "valid"
    df_test["subset"] = "test"

    df_train_valid_test = pd.concat([df_train, df_valid, df_test])

    return df_train_valid_test
