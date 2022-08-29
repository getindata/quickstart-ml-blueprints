import pandas as pd
import logging


log = logging.getLogger(__name__)

def train_val_split(df: pd.DataFrame, date_column: str, no_week: int = 1) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        date_column (str): _description_
        no_week (int, optional): _description_. Defaults to 1.

    Returns:
        pd.DataFrame: _description_
    """
    df[date_column] = pd.to_datetime(df[date_column])
    log.info(f'Dataframe size before splitting: {df.shape}')
    max_date = df[date_column].max()
    log.info(f'Dataframe min date: {df[date_column].min()}, max date: {max_date}')
    end_train_date = max_date - pd.Timedelta(weeks=no_week)
    end_val_date = end_train_date + pd.Timedelta(weeks=1)
    train_df = df[df[date_column]<=end_train_date]
    val_df = df[
        (df[date_column]>end_train_date) &
        (df[date_column]<=end_val_date)]
    log.info(f'Training dataframe size: {train_df.shape}, Validation dataframe size: {val_df.shape}')
    log.info(f'Training dataframe min date: {train_df[date_column].min()}, max date: {train_df[date_column].max()}')
    log.info(f'Validation dataframe min date: {val_df[date_column].min()}, max date: {val_df[date_column].max()}')
    return train_df, val_df
