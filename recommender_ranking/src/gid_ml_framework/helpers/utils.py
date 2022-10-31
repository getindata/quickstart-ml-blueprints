import functools
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def log_memory_usage(f):
    """Decorator function for logging memory usage of pd.DataFrame after executing function `f`"""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        df = f(*args, **kwargs)
        end_memory = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage after applying {f.__name__}: {end_memory:5.2f} MB")
        return df

    return wrapper


@log_memory_usage
def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Reduces memory usage for pd.DataFrame by changing to optimal numerical subtypes.

    Args:
        df (pd.DataFrame): dataframe

    Returns:
        pd.DataFrame: dataframe with optimized numerical column types
    """
    numerics = [
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        # pandas types
        "Int16",
        "Int32",
        "Int64",
        "Float16",
        "Float32",
        "Float64",
    ]
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).lower()[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        # if col_type == 'object':
    return df


def filter_dataframe_by_last_n_days(
    df: pd.DataFrame, n_days: int, date_column: str
) -> pd.DataFrame:
    """Filters out records in dataframe older than `max(date) - n_days`.

    Args:
        df (pd.DataFrame): dataframe with date column
        n_days (int): number of days to keep
        date_column (str): name of a column with date

    Returns:
        pd.DataFrame: filtered dataframe
    """
    if not n_days:
        logger.info("n_days is equal to None, skipping the filtering by date step.")
        return df
    try:
        df.loc[:, date_column] = pd.to_datetime(df.loc[:, date_column])
    except KeyError:
        logger.error("Given date_column does not exist in df")
        raise
    except ValueError:
        logger.error("Given date_column is not convertible to datetime")
        raise
    max_date = df.loc[:, date_column].max()
    filter_date = max_date - pd.Timedelta(days=n_days)
    logger.info(
        f"Maximum date is: {max_date}, date for filtering is: {filter_date}, {n_days=}"
    )
    logger.info(f"Shape before filtering: {df.shape}")
    df = df[df.loc[:, date_column] >= filter_date]
    logger.info(f"Shape after filtering: {df.shape}")
    return df
