import pandas as pd
import re
import logging
from typing import List, Any, Union
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

def add_label(candidates: pd.DataFrame, val_transactions: Union[pd.DataFrame, None]) -> pd.DataFrame:
    if val_transactions is None:
        logger.info('Skipping function add_label()')
        return candidates
    logger.info('Removing duplicate validations transactions')
    # duplicated transactions
    val_transactions = (
        val_transactions[['customer_id', 'article_id']]
            .drop_duplicates()
            .assign(label=lambda x: 1)
    )
    logger.info(f'Number of validation transactions left: {val_transactions.shape}')
    # sampling only some candidates for speed
    candidates = (
        candidates
            .sample(frac=1, random_state=888)
            .groupby(['customer_id'])
            .head(15)
            .reset_index(drop=True)
    )
    candidates = (
        candidates
            .merge(val_transactions, on=['customer_id', 'article_id'], how='left')
            .fillna({'label': 0})
    )
    candidates['label'] = candidates['label'].astype(int)
    logger.info(candidates.label.value_counts(normalize=True))
    logger.info(candidates.label.value_counts(normalize=False))
    return candidates

def add_article_features(
    candidates: pd.DataFrame,
    automated_articles_features: pd.DataFrame,
    manual_article_features: pd.DataFrame,
    regex_pattern: str
    ) -> pd.DataFrame:
    # # just for now
    # return candidates

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
    logger.info(f'Candidates dataframe shape after joining article features: {candidates.shape}')
    return candidates

def add_customer_features(
    candidates: pd.DataFrame,
    automated_customers_features: pd.DataFrame,
    manual_customer_features: pd.DataFrame,
    regex_pattern: str) -> pd.DataFrame:
    # # just for now
    # return candidates

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
    logger.info(f'Candidates dataframe shape after joining customer features: {candidates.shape}')
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