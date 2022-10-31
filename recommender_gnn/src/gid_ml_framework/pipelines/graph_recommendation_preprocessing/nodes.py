import logging
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd

from gid_ml_framework.extras.datasets.chunks_dataset import _concat_chunks

pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)


def concat_train_val(
    train_df: Iterator[pd.DataFrame], val_df: Iterator[pd.DataFrame], date_column: str
) -> pd.DataFrame:
    """Concatenate train and val transactions subsets for preprocessing purposes (there is no data leak from previous
    data imputation, because we are not using imputed data for following tasks). Also converts date column to timestamp
    and renames columns.

    Args:
        train_df (pd.DataFrame): transactions train dataframe
        val_df (pd.DataFrame): transaction val dataframe

    Returns:
        pd.DataFrame: concatenated transactions dataframe
    """
    if not isinstance(train_df, pd.DataFrame):
        train_df = _concat_chunks(train_df)
        val_df = _concat_chunks(val_df)
    concat_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    concat_df.loc[:, date_column] = pd.to_datetime(concat_df.loc[:, date_column])
    concat_df.loc[:, "time"] = (
        concat_df.loc[:, date_column].values.astype(np.int64) // 10**9
    )
    concat_df.drop(date_column, axis=1, inplace=True)
    concat_df.rename(
        columns={"article_id": "item_id", "customer_id": "user_id"}, inplace=True
    )
    logger.info(f"Concatenated transactions dataframe shape: {concat_df.shape}")
    return concat_df


def _create_mapping(df: pd.DataFrame, map_column: str) -> Dict:
    """Creates mapping into consecutive integers for given column."""
    ids = np.sort(df.loc[:, map_column].unique())
    mapping = {v: k for k, v in enumerate(ids)}
    return mapping


def map_users_and_items(
    transactions_df: Iterator[pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Map users and items ids to consecutive integers.

    Args:
        transactions_df (pd.DataFrame): concatenated train and val transactions dataframes

    Returns:
        Tuple: tuple of dataframes including original dataframe with mapping applied and mappings for users and items
    """
    if not isinstance(transactions_df, pd.DataFrame):
        transactions_df = _concat_chunks(transactions_df)
    logger.info(f"Transactions dataframe shape: {transactions_df.shape}")
    user_column = "user_id"
    item_column = "item_id"
    time_column = "time"
    users_mapping = _create_mapping(transactions_df, map_column=user_column)
    items_mapping = _create_mapping(transactions_df, map_column=item_column)
    transactions_df.loc[:, user_column] = transactions_df.loc[:, user_column].map(
        users_mapping.get
    )
    transactions_df.loc[:, item_column] = transactions_df.loc[:, item_column].map(
        items_mapping.get
    )
    transactions_df = transactions_df.loc[:, [user_column, item_column, time_column]]
    logger.info(
        f"Max and min user_ids: {max(transactions_df.loc[:, user_column])}, {min(transactions_df.loc[:, user_column])}"
    )
    logger.info(
        f"Max and min item_ids: {max(transactions_df.loc[:, item_column])}, {min(transactions_df.loc[:, item_column])}"
    )
    return (transactions_df, users_mapping, items_mapping)
