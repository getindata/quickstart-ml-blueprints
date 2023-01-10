"""This module contains common data preprocessing functionalities for different pipelines."""
import re
from typing import Tuple

import numpy as np
import pandas as pd


def extract_column_names(df: pd.DataFrame) -> Tuple[list, list, list, str]:
    """Extract info, numeric, categorical and target column names based on the column naming convention.

    Column naming convention based on prefixes:
    - 'i_*': informative columns
    - 'n_*': numeric columns
    - 'c_*': categorical columns
    - 'y_*': target column

    Args:
        df (pd.DataFrame): data frame to extract column names from

    Returns:
        Tuple[list, list, list, str]: tuple containing column names of different types in given order.
        The order is: info columns, numeric columns, categorical columns, target column.
    """
    info_cols = [item for item in df.columns if re.compile("^i_").match(item)]
    num_cols = [item for item in df.columns if re.compile("^n_").match(item)]
    cat_cols = [item for item in df.columns if re.compile("^c_").match(item)]
    target_col = [item for item in df.columns if re.compile("^y_").match(item)][0]

    return info_cols, num_cols, cat_cols, target_col


def ensure_column_types(
    df: pd.DataFrame, num_cols: list, cat_cols: list
) -> pd.DataFrame:
    """Force correct column types for categorical and numerical columns.

    Args:
        df (pd.DataFrame): input data frame
        num_cols (list): list of numerical column names
        cat_cols (list): list of categorical column names

    Returns:
        pd.DataFrame: data frame with correct column types
    """
    df[num_cols] = df[num_cols].astype(float)
    df[cat_cols] = np.where(
        pd.isnull(df[cat_cols]), df[cat_cols], df[cat_cols].astype(str)
    )

    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Get rid of symbols presenent in column names that are not accepted by some models.
    Those column names may appear after one-hot encoding.

    Args:
        df (pd.DataFrame): input data frame

    Returns:
        pd.DataFrame: data frame with corrected column names
    """
    df.columns = [re.sub(r"<|>|\[|\]", "", item) for item in df.columns]

    return df
