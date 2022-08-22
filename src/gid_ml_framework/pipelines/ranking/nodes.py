import pandas as pd
import numpy as np
import logging
import lightgbm
from typing import Any, List
import re
from ...helpers.utils import reduce_memory_usage, log_memory_usage


logger = logging.getLogger(__name__)

@log_memory_usage
def _fill_na_cast_to_int(df: pd.DataFrame, regex_pattern: str, fill_na_value: Any) -> pd.DataFrame:
    cols = [col for col in df.columns if re.match(regex_pattern, col)]
    df.loc[:, cols] = df.loc[:, cols].fillna(fill_na_value).astype(int)
    return df

@log_memory_usage
def _cast_as_category(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df.loc[:, cols] = df.loc[:, cols].astype('category')
    return df

def add_article_features(
    candidates: pd.DataFrame,
    automated_articles_features: pd.DataFrame,
    manual_article_features: pd.DataFrame,
    regex_pattern: str
    ) -> pd.DataFrame:
    # reduce memory
    candidates = reduce_memory_usage(candidates)
    automated_articles_features = reduce_memory_usage(automated_articles_features)
    manual_article_features = reduce_memory_usage(manual_article_features)
    # merge
    candidates = (
        candidates
            .merge(automated_articles_features, how='left', on='article_id')
            .merge(manual_article_features, how='left', on='article_id')
    )
    candidates = _fill_na_cast_to_int(candidates, regex_pattern, 0)
    candidates = reduce_memory_usage(candidates)
    return candidates

def add_customer_features(
    candidates: pd.DataFrame,
    automated_customers_features: pd.DataFrame,
    manual_customer_features: pd.DataFrame,
    regex_pattern: str
    ) -> pd.DataFrame:
    # reduce memory
    candidates = reduce_memory_usage(candidates)
    automated_customers_features = reduce_memory_usage(automated_customers_features)
    manual_customer_features = reduce_memory_usage(manual_customer_features)
    # merge
    candidates = (
        candidates
            .merge(automated_customers_features, how='left', on='customer_id')
            .merge(manual_customer_features, how='left', on='customer_id')
    )
    candidates = _fill_na_cast_to_int(candidates, regex_pattern, 0)
    candidates = reduce_memory_usage(candidates)
    return candidates

def add_dict_features(
    candidates: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    category_cols: List[str],
    drop_cols: List[str]
    ) -> pd.DataFrame:
    # reduce memory
    candidates = reduce_memory_usage(candidates)
    articles = reduce_memory_usage(articles)
    customers = reduce_memory_usage(customers)
    # merge/drop
    candidates = (
        candidates
            .merge(articles, how='left', on='article_id')
            .merge(customers, how='left', on='customer_id')
            .drop(drop_cols, axis=1)
    )
    candidates = _cast_as_category(candidates, category_cols)
    return candidates

def train_model(candidates: pd.DataFrame, val_transactions: pd.DataFrame):
    pass
