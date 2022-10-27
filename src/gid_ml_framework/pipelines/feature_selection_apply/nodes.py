import logging
from typing import Set
import pandas as pd


logger = logging.getLogger(__name__)

def apply_feature_selection(df: pd.DataFrame, columns: Set) -> pd.DataFrame:
    """Apply feature selection columns

    Args:
        df (pd.DataFrame): dataframe
        columns (Set): columns to keep

    Returns:
        pd.DataFrame: dataframe with selected columns
    """
    logger.info(f'Applying feature selection, number of columns before: {len(df.columns)}')
    df = df.loc[:, df.columns.isin(columns)]
    logger.info(f'Number of columns after feature selection: {len(df.columns)}')
    return df
