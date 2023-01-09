import logging

import pandas as pd

log = logging.getLogger(__name__)


def extract_transactions(
    preprocessed_df: pd.DataFrame,
    act_article_col: str = "article_id",
    act_customer_col: str = "customer_id",
    act_timestamp_col: str = "timestamp",
    original_article_col: str = "aid",
    original_customer_col: str = "session",
    original_timestamp_col: str = "ts",
    original_type_col: str = "type",
) -> pd.DataFrame:
    """Extracts all transactions data from otto input dataframe, also renames columns to act format.

    Args:
        preprocessed_df (pd.DataFrame): preprocessed otto dataframe
        act_article_col (str): name of article column in act format
        act_customer_col (str): name of customer column in act format
        act_timestamp_col (str): name of timestamp column in act format
        original_article_col (str): name of article column in original format
        original_customer_col (str): name of customer column in original format
        original_timestamp_col (str): name of timestamp column in original format
        original_type_col (str): name of type column in original format
    """
    preprocessed_df.loc[
        :, [original_article_col, original_type_col]
    ] = preprocessed_df.loc[:, [original_article_col, original_type_col]].astype(str)
    preprocessed_df.loc[:, original_article_col] = (
        preprocessed_df.loc[:, original_article_col]
        + "_"
        + preprocessed_df.loc[:, original_type_col]
    )
    transactions_df = preprocessed_df.drop(columns=[original_type_col])
    transactions_df.rename(
        columns={
            original_article_col: act_article_col,
            original_customer_col: act_customer_col,
            original_timestamp_col: act_timestamp_col,
        },
        inplace=True,
    )
    log.info(f"Number of transactions: {transactions_df.shape[0]}")
    return transactions_df


def _extract_entity(transactions_df: pd.DataFrame, act_entity_col: str) -> pd.DataFrame:
    entity_df = pd.DataFrame(
        {act_entity_col: transactions_df.loc[:, act_entity_col].unique()}
    )
    log.info(f"Number of unique {act_entity_col}: {entity_df.shape[0]}")
    return entity_df


def extract_articles(
    transactions_df: pd.DataFrame, act_article_col: str = "article_id"
) -> pd.DataFrame:
    """Extract all articles data from otto transactions dataframe. In case of otto dataset
    there are only article ids in transactions_df.

    Args:
        transactions_df (pd.DataFrame): transactions otto dataframe
        act_article_col (str): name of article column in act format
    """
    articles_df = _extract_entity(transactions_df, act_article_col)
    return articles_df


def extract_customers(
    transactions_df: pd.DataFrame, act_customer_col: str = "customer_id"
) -> pd.DataFrame:
    """Extract all customers data from otto transactions dataframe. In case of otto dataset
    there are only customer ids in input_df.

    Args:
        transactions_df (pd.DataFrame): transactions otto dataframe
        act_customer_col (str): name of customer column in act format
    """
    customers_df = _extract_entity(transactions_df, act_customer_col)
    return customers_df
