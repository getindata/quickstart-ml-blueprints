import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _stratify(
    input_otto_df: pd.DataFrame,
    sessions_frac: float,
    session_column: str = "session",
    timestamp_column: str = "ts",
    event_type_column: str = "type",
) -> List:
    """Stratify input dataframe based on length of sessions, their timestamps and
    event types (clicks, purchases, additions to cart)

    Args:
       input_otto_df (pd.DataFrame): otto dataframe
        session_frac (float): fraction of sessions to sample
        session_column (str): name of session column
        timestamp_column (str): name of timestamp column
        event_type_column (str): name of event type column

    Returns:
        List: stratified sample of sessions
    """
    stratify_columns = ["length", "mean_timestamp", "event_unique"]
    length_column, mean_timestamp_column, event_unique_column = stratify_columns
    grouped_df = input_otto_df.groupby(session_column, as_index=False).agg(
        length=(session_column, "count"),
        mean_timestamp=(timestamp_column, "mean"),
        event_unique=(event_type_column, "nunique"),
    )

    grouped_df.loc[:, length_column] = pd.qcut(
        grouped_df.loc[:, length_column], q=3, duplicates="drop"
    ).astype("category")
    grouped_df.loc[:, mean_timestamp_column] = pd.qcut(
        grouped_df.loc[:, mean_timestamp_column], q=5, duplicates="drop"
    ).astype("category")
    grouped_df.loc[:, event_unique_column] = pd.qcut(
        grouped_df.loc[:, event_unique_column], q=3, duplicates="drop"
    ).astype("category")

    _, sampled_df = train_test_split(
        grouped_df,
        test_size=sessions_frac,
        stratify=grouped_df.loc[:, stratify_columns],
    )
    sampled_ids = pd.Series(sampled_df.loc[:, session_column].unique())
    return sampled_ids


def sample(
    input_otto_df: pd.DataFrame,
    sessions_frac: float = 0.1,
    stratify: bool = False,
    session_column: str = "session",
    timestamp_column: str = "ts",
    event_type_column: str = "type",
) -> pd.DataFrame:
    """Sample Santader data based on customer sample size and cutoff date.

    Args:
       input_otto_df (Iterator[pd.DataFrame]): otto dataframe
        sample_customer_frac (float): fraction of sessions to sample
        stratify (Boolean): should sample be stratified based on length of sessions,
            their timestamps and event types (clicks, purchases, additions to cart)
        session_column (str): name of session column
        timestamp_column (str): name of timestamp column
        event_type_column (str): name of event type column

    Returns:
        pd.DataFrame: otto data sample
    """
    logger.info(f"Dataframe shape before sampling: {input_otto_df.shape}")
    if np.isclose(sessions_frac, 1.0):
        sampled_df = input_otto_df
    else:
        unique_ids = pd.Series(input_otto_df.loc[:, session_column].unique())
        sessions_limit = int(len(unique_ids) * sessions_frac)
        if not sessions_limit:
            return pd.DataFrame({})
        if stratify:
            sampled_ids = _stratify(
                input_otto_df,
                sessions_frac,
                session_column,
                timestamp_column,
                event_type_column,
            )
        else:
            sampled_ids = unique_ids.sample(sessions_limit)
        sampled_df = input_otto_df.loc[
            input_otto_df.loc[:, session_column].isin(sampled_ids)
        ]
    logger.info(f"Dataframe shape after sampling: {sampled_df.shape}")
    return sampled_df
