import logging
from typing import Dict, Set
import pandas as pd
import numpy as np
from featuretools.selection import (
    # remove_highly_correlated_features() does not work with booleans in current version of featuretools
    # remove_highly_correlated_features,
    remove_highly_null_features,
    remove_single_value_features)


logger = logging.getLogger(__name__)

def _remove_correlated_features(df: pd.DataFrame, corr_threshold: float = 0.99) -> pd.DataFrame:
    """Given a dataframe and a correlation threshold (absolute), removes all but one correlated features.

    Args:
        df (pd.DataFrame): dataframe
        corr_threshold (float, optional): absolute correlation threshold for removing feature. Defaults to 0.99.

    Returns:
        pd.DataFrame: dataframe without correlated features
    """
    # corr for numerical cols
    corr_matrix = df.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    # iteration
    col_list = df.columns.to_list()
    correlated_columns = set()
    non_correlated_columns = set()
    for col in col_list:
        if col in correlated_columns:
            continue
        non_correlated_columns.add(col)
        corr_cols_list = corr_matrix.index[corr_matrix.loc[:, col].ge(corr_threshold)].to_list()
        if len(corr_cols_list)>0:
            correlated_columns |= set(corr_cols_list)
    logger.info(f'Correlated: {len(correlated_columns)=}, \n Uncorrelated: {len(non_correlated_columns)=}, \n All: {len(col_list)=}')
    assert len(correlated_columns)+len(non_correlated_columns)==len(col_list)
    df = df.drop(list(correlated_columns), axis=1)
    logger.info(f'Number of correlated features: {len(correlated_columns)}')
    return df

def feature_selection(df: pd.DataFrame, selection_params: Dict, feature_selection: bool) -> Set:
    """Applies multiple feature_selection functions to a dataframe: highly null values, single value features, correlated features.

    Args:
        df (pd.DataFrame): dataframe
        selection_params (Dict): parameters for feature_selection functions
        feature_selection (bool): whether to apply feature selection to a given dataframe

    Returns:
        Set: selected features
    """
    if not feature_selection:
        logger.info(f'feature_selection is {feature_selection} -> not applying any feature selection functions to a dataframe')
        return set(df.columns)
    initial_cols = set(df.columns)
    logger.info(f'Shape before feature selection: {df.shape}')
    df = remove_highly_null_features(df, pct_null_threshold=selection_params['pct_null_threshold'])
    df = remove_single_value_features(df)
    df = _remove_correlated_features(df, corr_threshold=selection_params['corr_threshold'])
    logger.info(f'Shape after feature selection: {df.shape}')
    feature_selection_cols = set(df.columns)
    logger.info(f'Number of removed cols: {len(initial_cols.difference(feature_selection_cols))}')
    return feature_selection_cols
