import logging
from typing import Dict, Iterator, Tuple, Union

import numpy as np
import pandas as pd

from recommender_gnn.extras.datasets.chunks_dataset import _concat_chunks

pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)


def preprocess_transactions(
    train_df: Union[Iterator[pd.DataFrame], pd.DataFrame],
    val_df: Union[Iterator[pd.DataFrame], pd.DataFrame],
    train_subset: bool,
    val_subset: bool,
    original_date_column: str = "date",
    original_item_column: str = "article_id",
    original_user_column: str = "customer_id",
    new_date_column: str = "time",
    new_item_column: str = "item_id",
    new_user_column: str = "user_id",
) -> Union[pd.DataFrame, None]:
    """If both present concatenates train and val transactions subsets for preprocessing purposes (there is no data leak
    from previous data imputation, because we are not using imputed data for following tasks). Also converts date column
    to timestamp and renames columns.

    Args:
        train_df (pd.DataFrame): transactions train dataframe
        val_df (pd.DataFrame): transaction val dataframe
        train_subset (bool): whether to include train subset in preprocessing
        val_subset (bool): whether to include val subset in preprocessing
        original_date_column (str): original date column name
        original_item_column (str): original item column name
        original_user_column (str): original user column name
        new_date_column (str): new date column name
        new_item_column (str): new item column name
        new_user_column (str): new user column name

    Returns:
        pd.DataFrame: concatenated transactions dataframe
    """
    train_df = _concat_chunks(train_df) if train_subset else None
    val_df = _concat_chunks(val_df) if val_subset else None
    df = pd.concat([train_df, val_df]).reset_index(drop=True)
    if df is None:
        return df
    if original_date_column == "date":
        df.loc[:, original_date_column] = pd.to_datetime(
            df.loc[:, original_date_column]
        )
        df.loc[:, original_date_column] = (
            df.loc[:, original_date_column].values.astype(np.int64) // 10**9
        )
    df.rename(
        columns={
            original_date_column: new_date_column,
            original_item_column: new_item_column,
            original_user_column: new_user_column,
        },
        inplace=True,
    )
    logger.info(f"Preprocessed transactions dataframe shape: {df.shape}")
    return df


def _create_mapping(df: pd.DataFrame, map_column: str) -> Dict:
    """Creates mapping into consecutive integers for a given column."""
    ids = np.sort(df.loc[:, map_column].unique())
    mapping = {v: k for k, v in enumerate(ids)}
    return mapping


def map_users_and_items(
    transactions_df: Iterator[pd.DataFrame],
    user_column: str = "user_id",
    item_column: str = "item_id",
    time_column: str = "time",
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Map users and items ids to consecutive integers.

    Args:
        transactions_df (pd.DataFrame): concatenated train and val transactions dataframes

    Returns:
        Tuple: tuple of dataframes including original dataframe with mapping applied and mappings for users and items
    """
    transactions_df = _concat_chunks(transactions_df)
    logger.info(f"Transactions dataframe shape: {transactions_df.shape}")
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
