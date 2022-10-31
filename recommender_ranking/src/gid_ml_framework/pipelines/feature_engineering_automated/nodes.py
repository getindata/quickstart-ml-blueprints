import logging
from typing import List, Optional, Tuple

import featuretools as ft
import pandas as pd
from woodwork.logical_types import (
    AgeNullable,
    Boolean,
    Categorical,
    Double,
    NaturalLanguage,
)

from gid_ml_framework.helpers.utils import filter_dataframe_by_last_n_days

logger = logging.getLogger(__name__)


def _create_entity_set(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    n_days: int,
) -> ft.EntitySet:
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

    transactions = filter_dataframe_by_last_n_days(
        transactions, n_days, date_column="t_dat"
    )
    # transactions
    # needs unique (!) index
    transactions["_index"] = transactions.index
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
            "age": AgeNullable,
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

    logger.info(f"Initialized EntitySet: {es} \n\n for {n_days=}")
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
        agg_primitives=["sum", "mean", "median", "max", "min", "count"],
    )
    # keep only transformations
    logger.info(
        f"Number of articles features before dropping non-transformations: {len(articles_feature_defs)}"
    )
    articles_feats = [
        feature.get_name()
        for feature in articles_feature_defs
        if len(feature.base_features) > 0
    ]
    articles_feature_matrix = articles_feature_matrix[articles_feats]
    logger.info(
        f"Number of articles features after dropping non-transformations: {len(articles_feats)}"
    )
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
    logger.info(
        f"Number of customer features before dropping non-transformations: {len(customers_feature_defs)}"
    )
    customer_feats = [
        feature.get_name()
        for feature in customers_feature_defs
        if len(feature.base_features) > 0
    ]
    customers_feature_matrix = customers_feature_matrix[customer_feats]
    logger.info(
        f"Number of customer features after dropping non-transformations: {len(customer_feats)}"
    )
    return customers_feature_matrix


def create_static_features(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    n_days_list: List[Optional[int]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Creates static features for articles and customers based on different time windows (`n_days_list`).

    Args:
        transactions (pd.DataFrame): transactions
        customers (pd.DataFrame): customers
        articles (pd.DataFrame): articles
        n_days_list (List[Optional[int]]): list of different time windows for feature calculation

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
        tuple constisting of pd.DataFrames with static articles and static customers respecitively
    """
    articles_dataframes_list, customers_dataframes_list = list(), list()
    for n_days in n_days_list:
        suffix_str_article = (
            "_all_articles_automated"
            if n_days is None
            else f"_{n_days}_articles_automated"
        )
        suffix_str_customer = (
            "_all_customers_automated"
            if n_days is None
            else f"_{n_days}_customers_automated"
        )
        es = _create_entity_set(transactions, customers, articles, n_days)
        # articles
        articles_feature_matrix = _create_static_articles_features(es)
        articles_feature_matrix = articles_feature_matrix.add_suffix(suffix_str_article)
        articles_dataframes_list.append(articles_feature_matrix)
        # customers
        customers_feature_matrix = _create_static_customer_features(es)
        customers_feature_matrix = customers_feature_matrix.add_suffix(
            suffix_str_customer
        )
        customers_dataframes_list.append(customers_feature_matrix)
    static_articles = pd.concat(articles_dataframes_list, axis=1)
    static_customers = pd.concat(customers_dataframes_list, axis=1)
    return static_articles, static_customers
