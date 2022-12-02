import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _stratify(
    input_df: pd.DataFrame, sessions_frac: float, session_column: str
) -> List:
    """Stratify input dataframe based on length of sessions, their timestamps and
    event types (clicks, purchases, additions to cart)

    Args:
        input_df (pd.DataFrame): otto dataframe
        session_frac (float): fraction of sessions to sample
        session_column (str): name of session column

    Returns:
        List: stratified sample of sessions
    """
    timestamp_column = "ts"
    event_type_column = "type"
    stratify_columns = ["length", "mean_timestamp", "event_type"]
    length_column, mean_timestamp_column, event_type_column = stratify_columns
    grouped_df = input_df.groupby(session_column).agg(
        length=(session_column, "count"),
        mean_timestamp=(timestamp_column, "mean"),
        event_type=(event_type_column, "nunique"),
    )

    grouped_df.loc[:, length_column] = pd.qcut(
        grouped_df.loc[:, length_column], q=3, duplicates="drop"
    ).astype("category")
    grouped_df.loc[:, mean_timestamp_column] = pd.qcut(
        grouped_df.loc[:, mean_timestamp_column], q=5, duplicates="drop"
    ).astype("category")
    grouped_df.loc[:, event_type_column] = pd.qcut(
        grouped_df.loc[:, event_type_column], q=3, duplicates="drop"
    ).astype("category")

    _, sampled_df = train_test_split(
        grouped_df,
        test_size=sessions_frac,
        stratify=grouped_df.loc[:, stratify_columns],
    )
    sampled_ids = pd.Series(sampled_df.loc[:, session_column].unique())
    return sampled_ids


def sample(
    input_df: pd.DataFrame,
    sessions_frac: float = 0.1,
    stratify: bool = False,
) -> pd.DataFrame:
    """Sample Santader data based on customer sample size and cutoff date.

    Args:
        input_df (Iterator[pd.DataFrame]): otto dataframe
        sample_customer_frac (float): fraction of sessions to sample
        stratify (Boolean): should sample be stratified based on length of sessions,
            their timestamps and event types (clicks, purchases, additions to cart)

    Returns:
        pd.DataFrame: otto data sample
    """
    logger.info(f"Dataframe shape before sampling: {input_df.shape}")
    session_column = "session"
    if np.isclose(sessions_frac, 1.0):
        sampled_df = input_df
    else:
        unique_ids = pd.Series(input_df.loc[:, session_column].unique())
        sessions_limit = int(len(unique_ids) * sessions_frac)
        if not sessions_limit:
            return pd.DataFrame({})
        if stratify:
            sampled_ids = _stratify(input_df, sessions_frac, session_column)
        else:
            sampled_ids = unique_ids.sample(sessions_limit)
        sampled_df = input_df.loc[input_df.loc[:, session_column].isin(sampled_ids)]
    logger.info(f"Dataframe shape after sampling: {sampled_df.shape}")
    return sampled_df
