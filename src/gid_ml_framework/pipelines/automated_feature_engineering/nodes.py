import logging
from typing import List, Tuple, Callable
import pandas as pd
import featuretools as ft
from woodwork.logical_types import Categorical, Boolean, AgeNullable, Double, NaturalLanguage
from featuretools.selection import (
    # remove_highly_correlated_features() does not work with Booleans in current version of featuretools
    # remove_highly_correlated_features,
    remove_highly_null_features,
    remove_single_value_features,
)


logger = logging.getLogger(__name__)

def create_entity_set(transactions: pd.DataFrame, customers: pd.DataFrame, articles: pd.DataFrame) -> ft.EntitySet:
    es = ft.EntitySet(id="kaggle_hm_data")

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
    logger.info(f'Initialized EntitySet: {es}')
    return es

def _apply_feature_selection_fn(feature_matrix: pd.DataFrame, features: List, feature_selection_fn: Callable, **kwargs: float) -> Tuple[pd.DataFrame, List]:
    old_features_set = set(features)
    logger.info(f'Shape before applying {feature_selection_fn.__name__}: {feature_matrix.shape}')
    if kwargs:
        logger.info(f'keyword arguments: {kwargs}')
    new_feature_matrix, new_features = feature_selection_fn(feature_matrix, features=features, **kwargs)
    logger.info(f'Removed {len(old_features_set)-len(set(new_features))} features')
    logger.info(f'Columns: {old_features_set-set(new_features)}')
    logger.info(f'Shape after applying {feature_selection_fn.__name__}: {new_feature_matrix.shape}')
    return new_feature_matrix, new_features


def create_static_articles_features(es: ft.EntitySet, feature_selection: Boolean, null_threshold: float=0.9) -> pd.DataFrame:
    articles_feature_matrix, articles_feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="articles",
        agg_primitives=["sum", "mean", "median", "max", "min", "time_since_last", "count", "time_since_first"]
    )
    if not feature_selection:
        return articles_feature_matrix
    articles_feature_matrix, articles_feature_defs = _apply_feature_selection_fn(articles_feature_matrix, articles_feature_defs, remove_single_value_features)
    articles_feature_matrix, articles_feature_defs = _apply_feature_selection_fn(articles_feature_matrix, articles_feature_defs, remove_highly_null_features, pct_null_threshold=null_threshold) 
    return articles_feature_matrix

def create_static_customer_features(es: ft.EntitySet, feature_selection: Boolean, null_threshold: float=0.9) -> pd.DataFrame:
    customers_feature_matrix, customers_feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="customers",
        ignore_dataframes=["articles"],
        agg_primitives=["sum", "mean", "median", "max", "min", "time_since_last", "count", "time_since_first", "avg_time_between"],
    )
    if not feature_selection:
        return customers_feature_matrix
    customers_feature_matrix, customers_feature_defs = _apply_feature_selection_fn(customers_feature_matrix, customers_feature_defs, remove_single_value_features)
    customers_feature_matrix, customers_feature_defs = _apply_feature_selection_fn(customers_feature_matrix, customers_feature_defs, remove_highly_null_features, pct_null_threshold=null_threshold)
    return customers_feature_matrix
