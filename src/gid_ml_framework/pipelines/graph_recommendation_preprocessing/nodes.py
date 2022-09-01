import logging
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd
from kedro.extras.datasets.pandas import CSVDataSet

from gid_ml_framework.extras.datasets.chunks_dataset import (
    _concat_chunks,
    _load,
)

pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)
# Overwriting load method because of chunksize bug in Kedro < 0.18
CSVDataSet._load = _load


def concat_train_val(
    train_df: Iterator[pd.DataFrame], val_df: Iterator[pd.DataFrame], date_column: str
) -> pd.DataFrame:
    """Concatenate train and val transactions subsets for preprocessing purposes. Also converts date column to timestamp
    and renames columns.

    Args:
        train_df (pd.DataFrame): transactions train dataframe
        val_df (pd.DataFrame): transaction val dataframe

    Returns:
        pd.DataFrame: concatenated transactions dataframe
    """
    train_df = _concat_chunks(train_df)
    val_df = _concat_chunks(val_df)
    concat_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    concat_df.loc[:, "time"] = (
        concat_df.loc[:, date_column].values.astype(np.int64) // 10**9
    )
    concat_df.drop(date_column, axis=1, inplace=True)
    concat_df.rename(
        columns={"articles_id": "item_id", "customer_id": "user_id"}, inplace=True
    )
    return concat_df


def _create_mapping(df: pd.DataFrame, map_column: str) -> Dict:
    """Creates mapping into consecutive integers for given column."""
    ids = df.loc[:, map_column].sort_values().reset_index(drop=True)
    mapping = {v: k for k, v in enumerate(ids)}
    return mapping


def map_users_and_items(
    transactions_df: Iterator[pd.DataFrame],
    customers_df: pd.DataFrame,
    articles_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Map users and items ids to consecutive integers.

    Args:
        transactions_df (pd.DataFrame): concatenated train and val transactions dataframes

    Returns:
        Tuple: tuple of dataframes including original dataframe with mapping applied and mappings for users and items
    """
    transactions_df = _concat_chunks(transactions_df)
    customers_df = _concat_chunks(customers_df)
    articles_df = _concat_chunks(articles_df)
    logger.info(f"Transactions dataframe shape: {transactions_df.shape}")
    users_mapping = _create_mapping(customers_df, map_column="customer_id")
    items_mapping = _create_mapping(articles_df, map_column="article_id")
    transactions_df.replace(
        {"user_id": users_mapping, "item_id": items_mapping}, inplace=True
    )
    return (transactions_df, users_mapping, items_mapping)
