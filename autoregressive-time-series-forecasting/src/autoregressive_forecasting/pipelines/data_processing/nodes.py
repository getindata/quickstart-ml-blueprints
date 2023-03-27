import logging

import pandas as pd
from autoregressive_forecasting.helpers.utils import filter_dataframe_by_date_cutoffs
from autoregressive_forecasting.helpers.utils import rename_columns


logger = logging.getLogger(__name__)


def _sum_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Sums sales for each store and date.

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: sales per ["unique_id", "ds"]
    """
    store_sales = df.groupby(["unique_id", "ds"])["y"].sum().reset_index()
    return store_sales


def preprocess_data(
    df: pd.DataFrame,
    mapper: dict[str, str],
    min_date_cutoff: str | None,
    max_date_cutoff: str | None,
) -> pd.DataFrame:
    """Prepares data for training.

    Args:
        df (pd.DataFrame): input dataframe
        mapper (dict[str, str]): mapper from raw dataframe column names to the column names we want
        min_date_cutoff (str | None): minimum date to include in the filtered dataframe
        max_date_cutoff (str | None): maximum date to include in the filtered dataframe

    Returns:
        pd.DataFrame: preprocessed dataframe
    """
    df_renamed = rename_columns(df, mapper)
    df_filtered = filter_dataframe_by_date_cutoffs(
        df_renamed, min_date_cutoff, max_date_cutoff
    )
    df_sum = _sum_sales(df_filtered)
    return df_sum
