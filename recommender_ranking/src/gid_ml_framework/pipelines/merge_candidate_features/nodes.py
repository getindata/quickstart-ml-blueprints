import logging
import re
from typing import Any, List, Union

import pandas as pd

from gid_ml_framework.helpers.utils import (
    log_memory_usage,
    reduce_memory_usage,
)

logger = logging.getLogger(__name__)


@log_memory_usage
def _fill_na_cast_to_int(
    df: pd.DataFrame, regex_pattern: str, fill_na_value: Any
) -> pd.DataFrame:
    """Fills NA values with `fill_na_value` in a dataframe for columns that match `regex_pattern`.

    Args:
        df (pd.DataFrame): dataframe
        regex_pattern (str): regex pattern
        fill_na_value (Any): fill value

    Returns:
        pd.DataFrame: dataframe with filled missing values
    """
    cols = [col for col in df.columns if re.match(regex_pattern, col)]
    df.loc[:, cols] = df.loc[:, cols].fillna(fill_na_value).astype(int)
    return df


@log_memory_usage
def _cast_as_category(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Casts lists of columns in a dataframe to `pd.Category`.

    Args:
        df (pd.DataFrame): dataframe
        cols (List[str]): list of column names

    Returns:
        pd.DataFrame: dataframe with categorical columns
    """
    df.loc[:, cols] = df.loc[:, cols].astype("category")
    return df


def add_label(
    candidates: pd.DataFrame, val_transactions: Union[pd.DataFrame, None]
) -> pd.DataFrame:
    """Adds label to candidates dataframe for training,
    or returns the same dataframe if there are no validation transactions.

    Args:
        candidates (pd.DataFrame): candidates
        val_transactions (Union[pd.DataFrame, None]): validation transactions

    Returns:
        pd.DataFrame: candidates dataframe with label (or without)
    """
    if val_transactions is None:
        logger.info("Skipping function add_label()")
        return candidates
    logger.info("Removing duplicate validations transactions")
    # duplicated transactions
    val_transactions = (
        val_transactions[["customer_id", "article_id"]]
        .drop_duplicates()
        .assign(label=lambda x: 1)
    )
    logger.info(f"Number of validation transactions left: {val_transactions.shape}")

    # # sampling only some candidates for speed
    # candidates = (
    #     candidates
    #         .sample(frac=1, random_state=888)
    #         .groupby(['customer_id'])
    #         .head(150)
    #         .reset_index(drop=True)
    # )

    candidates = candidates.merge(
        val_transactions, on=["customer_id", "article_id"], how="left"
    ).fillna({"label": 0})
    candidates["label"] = candidates["label"].astype(int)
    logger.info(candidates.label.value_counts(normalize=True))
    logger.info(candidates.label.value_counts(normalize=False))
    return candidates


def add_article_features(
    candidates: pd.DataFrame,
    automated_articles_features: pd.DataFrame,
    manual_article_features: pd.DataFrame,
    regex_pattern: str,
) -> pd.DataFrame:
    """Adds article features (automated & manual) to candidates dataframe based on `article_id`.

    Args:
        candidates (pd.DataFrame): candidates
        automated_articles_features (pd.DataFrame): automated articles features
        manual_article_features (pd.DataFrame): manual articles features
        regex_pattern (str): regex pattern for filling missing values

    Returns:
        pd.DataFrame: candidates with articles features
    """
    # # if testing
    # return candidates

    # reduce memory
    candidates = reduce_memory_usage(candidates)
    automated_articles_features = reduce_memory_usage(automated_articles_features)
    manual_article_features = reduce_memory_usage(manual_article_features)
    logger.info(
        f"Automated articles features shape: {automated_articles_features.shape}"
    )
    logger.info(f"Manual articles features shape: {manual_article_features.shape}")
    # merge
    candidates = candidates.merge(
        automated_articles_features, how="left", on="article_id"
    ).merge(manual_article_features, how="left", on="article_id")
    candidates = _fill_na_cast_to_int(candidates, regex_pattern, 0)
    candidates = reduce_memory_usage(candidates)
    logger.info(
        f"Candidates dataframe shape after joining article features: {candidates.shape}"
    )
    return candidates


def add_customer_features(
    candidates: pd.DataFrame,
    automated_customers_features: pd.DataFrame,
    manual_customer_features: pd.DataFrame,
    regex_pattern: str,
) -> pd.DataFrame:
    """Adds customer features (automated & manual) to candidates dataframe based on `customer_id`.

    Args:
        candidates (pd.DataFrame): candidates
        automated_customers_features (pd.DataFrame): automated customers features
        manual_customer_features (pd.DataFrame): manual customers features
        regex_pattern (str): regex pattern for filling missing values

    Returns:
        pd.DataFrame: candidates dataframe with customers features
    """
    # # if testing
    # return candidates

    # reduce memory
    candidates = reduce_memory_usage(candidates)
    automated_customers_features = reduce_memory_usage(automated_customers_features)
    manual_customer_features = reduce_memory_usage(manual_customer_features)
    logger.info(
        f"Automated customers features shape: {automated_customers_features.shape}"
    )
    logger.info(f"Manual customers features shape: {manual_customer_features.shape}")
    # merge
    candidates = candidates.merge(
        automated_customers_features, how="left", on="customer_id"
    ).merge(manual_customer_features, how="left", on="customer_id")
    candidates = _fill_na_cast_to_int(candidates, regex_pattern, 0)
    candidates = reduce_memory_usage(candidates)
    logger.info(
        f"Candidates dataframe shape after joining customer features: {candidates.shape}"
    )
    return candidates


def add_dict_features(
    candidates: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    category_cols: List[str],
    drop_cols: List[str],
) -> pd.DataFrame:
    """Adds dictionary features to candidates dataframe.

    Args:
        candidates (pd.DataFrame): candidates
        articles (pd.DataFrame): articles dictionary
        customers (pd.DataFrame): customers dictionary
        category_cols (List[str]): list of categorical columns
        drop_cols (List[str]): list of columns to drop

    Returns:
        pd.DataFrame: candidates dataframe with dictionary features
    """
    # reduce memory
    candidates = reduce_memory_usage(candidates)
    articles = reduce_memory_usage(articles)
    customers = reduce_memory_usage(customers)
    logger.info(f"Dictionary articles features shape: {articles.shape}")
    logger.info(f"Dictionary customers features shape: {customers.shape}")

    # merge/drop
    candidates = (
        candidates.merge(articles, how="left", on="article_id")
        .merge(customers, how="left", on="customer_id")
        .drop(drop_cols, axis=1)
    )
    candidates = _cast_as_category(candidates, category_cols)
    return candidates
