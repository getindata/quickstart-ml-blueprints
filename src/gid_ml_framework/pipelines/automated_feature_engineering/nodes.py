import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import featuretools as ft
from woodwork.logical_types import Categorical, Boolean, AgeNullable, Double, NaturalLanguage
from featuretools.selection import (
    # remove_highly_correlated_features() does not work with Booleans in current version of featuretools
    # remove_highly_correlated_features,
    remove_highly_null_features,
    remove_single_value_features)


logger = logging.getLogger(__name__)

def _filter_dataframe_by_last_n_days(df: pd.DataFrame, n_days: int, date_column: str) -> pd.DataFrame:
    """Filters out records in dataframe older than `max(date) - n_days`.

    Args:
        df (pd.DataFrame): dataframe with date column
        n_days (int): number of days to keep
        date_column (str): name of a column with date

    Returns:
        pd.DataFrame: filtered dataframe
    """
    if not n_days:
        logger.info(f'n_days is equal to None, skipping the filtering by date step.')
        return df
    df.loc[:, date_column] = pd.to_datetime(df.loc[:, date_column])
    max_date = df.loc[:, date_column].max()
    filter_date = max_date - pd.Timedelta(days=n_days)
    logger.info(f'Maximum date is: {max_date}, date for filtering is: {filter_date}, {n_days=}')
    logger.info(f'Shape before filtering by date: {df.shape}')
    df = df[df.loc[:, date_column]>=filter_date]
    logger.info(f'Shape after filtering by date: {df.shape}')
    return df

def _create_entity_set(transactions: pd.DataFrame, customers: pd.DataFrame, articles: pd.DataFrame, n_days: int) -> ft.EntitySet:
    """Creates a ft.EntitySet based on transactions, customers and articles and relationships between those tables.
    Before creating, it also filters transactions to latest `n_days`.

    Args:
        transactions (pd.DataFrame): transactions
        customers (pd.DataFrame): customers
        articles (pd.DataFrame): articles
        n_days (int): number of latest days for filtering transactions

    Returns:
        ft.EntitySet: data and typing information for the whole dataset
    """
    es = ft.EntitySet(id=f"kaggle_hm_data_{n_days}")

    transactions = _filter_dataframe_by_last_n_days(transactions, n_days, date_column='t_dat')
    # transactions
    # needs unique (!) index
    transactions['_index'] = transactions.index
    es = es.add_dataframe(
        dataframe_name="transactions",
        dataframe=transactions,
        index="_index",
        time_index="t_dat",
        logical_types={
            "customer_id": Categorical,
            "article_id": Categorical,
            "price": Double,
            "sales_channel_id": Categorical,
        },
    )

    # customers
    es = es.add_dataframe(
        dataframe_name="customers",
        dataframe=customers,
        index="customer_id",
        logical_types={
            "customer_id": Categorical,
            "FN": Boolean,
            "Active": Boolean,
            "club_member_status": Categorical,
            "fashion_news_frequency": Categorical,
            "age": AgeNullable
        },
    )

    # articles
    es = es.add_dataframe(
        dataframe_name="articles",
        dataframe=articles,
        index="article_id",
        logical_types={
            "article_id": Categorical,
            "detail_desc": NaturalLanguage,
        },
    )
    
    # add relationships
    es = es.add_relationship("customers", "customer_id", "transactions", "customer_id")
    es = es.add_relationship("articles", "article_id", "transactions", "article_id")

    logger.info(f'Initialized EntitySet: {es} \n\n for {n_days=}')
    return es

def _create_static_articles_features(es: ft.EntitySet) -> pd.DataFrame:
    """Creates static article features not including the base features from articles.

    Args:
        es (ft.EntitySet): data and typing information for the whole dataset

    Returns:
        pd.DataFrame: static article features
    """
    articles_feature_matrix, articles_feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="articles",
        agg_primitives=["sum", "mean", "median", "max", "min", "count"]
    )
    # keep only transformations
    logger.info(f'Number of articles features before dropping non-transformations: {len(articles_feature_defs)}')
    articles_feats = [feature.get_name() for feature in articles_feature_defs if len(feature.base_features)>0]
    articles_feature_matrix = articles_feature_matrix[articles_feats]
    logger.info(f'Number of articles features after dropping non-transformations: {len(articles_feats)}')
    return articles_feature_matrix

def _create_static_customer_features(es: ft.EntitySet) -> pd.DataFrame:
    """Creates static customers features not including the base features from customers.

    Args:
        es (ft.EntitySet): data and typing information for the whole dataset

    Returns:
        pd.DataFrame: static customer features
    """
    customers_feature_matrix, customers_feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="customers",
        ignore_dataframes=["articles"],
        agg_primitives=["sum", "mean", "median", "max", "min", "count"],
    )
    # keep only transformations
    logger.info(f'Number of customer features before dropping non-transformations: {len(customers_feature_defs)}')
    customer_feats = [feature.get_name() for feature in customers_feature_defs if len(feature.base_features)>0]
    customers_feature_matrix = customers_feature_matrix[customer_feats]
    logger.info(f'Number of customer features after dropping non-transformations: {len(customer_feats)}')
    return customers_feature_matrix

def create_static_features(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    n_days_list: List[Optional[int]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Creates static features for articles and customers based on different time windows (`n_days_list`).

    Args:
        transactions (pd.DataFrame): transactions
        customers (pd.DataFrame): customers
        articles (pd.DataFrame): articles
        n_days_list (List[Optional[int]]): list of different time windows for feature calculation

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: tuple constisting of pd.DataFrames with static articles and static customers respecitively
    """
    articles_dataframes_list, customers_dataframes_list = list(), list()
    for n_days in n_days_list:
        suffix_str_article = '_all_articles' if n_days is None else f'_{n_days}_articles'
        suffix_str_customer = '_all_customers' if n_days is None else f'_{n_days}_customers'
        es = _create_entity_set(transactions, customers, articles, n_days)
        # articles
        articles_feature_matrix = _create_static_articles_features(es)
        articles_feature_matrix = articles_feature_matrix.add_suffix(suffix_str_article)
        articles_dataframes_list.append(articles_feature_matrix)
        # customers
        customers_feature_matrix = _create_static_customer_features(es)
        customers_feature_matrix = customers_feature_matrix.add_suffix(suffix_str_customer)
        customers_dataframes_list.append(customers_feature_matrix)
    static_articles = pd.concat(articles_dataframes_list, axis=1)
    static_customers = pd.concat(customers_dataframes_list, axis=1)
    return static_articles, static_customers

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

def feature_selection(df: pd.DataFrame, feature_selection: Boolean, selection_params: Dict) -> pd.DataFrame:
    """Applies multiple feature_selection functions to a dataframe: highly null values, single value features, correlated features.

    Args:
        df (pd.DataFrame): dataframe
        feature_selection (Boolean): whether to apply feature selection to a given dataframe
        selection_params (Dict): parameters for feature_selection functions

    Returns:
        pd.DataFrame: dataframe with selected features
    """
    if not feature_selection:
        logger.info(f'feature_selection is {feature_selection} -> not applying any feature selection functions to dataframe')
        return df
    logger.info(f'Shape before feature selection: {df.shape}')
    df = remove_highly_null_features(df, pct_null_threshold=selection_params['pct_null_threshold'])
    df = remove_single_value_features(df)
    df = _remove_correlated_features(df, corr_threshold=selection_params['corr_threshold'])
    logger.info(f'Shape after feature selection: {df.shape}')
    return df
