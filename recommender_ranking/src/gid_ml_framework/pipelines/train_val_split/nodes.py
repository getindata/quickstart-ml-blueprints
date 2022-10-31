import logging

import pandas as pd

logger = logging.getLogger(__name__)


def train_val_split(
    df: pd.DataFrame, date_column: str, no_train_weeks: int = 6, no_val_weeks: int = 1
) -> pd.DataFrame:
    """Splits dataframe into training and validation datasets.
    Validation dates are between: max(date_column) and max(date_column)-no_val_weeks
    Training dates are between: max(date_column)-no_val_weeks and max(date_column)-(no_val_weeks+no_train_weeks)

    Args:
        df (pd.DataFrame): dataframe
        date_column (str): column name with date
        no_train_week (int, optional): number of weeks for training dataset. Defaults to 6.
        no_val_week (int, optional): number of weeks for validation dataset. Defaults to 1.

    Returns:
        pd.DataFrame: tuple of training and validation dataframes respectively
    """
    try:
        df.loc[:, date_column] = pd.to_datetime(df.loc[:, date_column])
    except KeyError:
        logger.error("Given date_column does not exist in df")
        raise
    except ValueError:
        logger.error("Given date_column is not convertible to datetime")
        raise
    logger.info(f"Dataframe size before splitting: {df.shape}")
    max_date = df.loc[:, date_column].max()
    logger.info(f"Dataframe min date: {df[date_column].min()}, max date: {max_date}")
    end_train_date = max_date - pd.Timedelta(weeks=no_val_weeks)
    start_train_date = end_train_date - pd.Timedelta(weeks=no_train_weeks)
    train_df = df[
        (df[date_column] > start_train_date) & (df[date_column] <= end_train_date)
    ]
    val_df = df[(df[date_column] > end_train_date)]
    logger.info(
        f"Training dataframe size: {train_df.shape}, Validation dataframe size: {val_df.shape}"
    )
    logger.info(
        f"Training dataframe min date: {train_df[date_column].min()}, max date: {train_df[date_column].max()}"
    )
    logger.info(
        f"Validation dataframe min date: {val_df[date_column].min()}, max date: {val_df[date_column].max()}"
    )
    return train_df, val_df
