import logging
import pandas as pd
from typing import List, Optional, Union


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

def _concat_dataframes_on_index(list_of_dfs: List[pd.DataFrame], index_name: Union[str, List[str]]) -> pd.DataFrame:
    """Concatenates multiple dataframes on index.

    Args:
        list_of_dfs (List[pd.DataFrame]): dataframes with index_name as column
        index_name (Union[str, List[str]]): column name(s) on which to merge dataframes

    Returns:
        pd.DataFrame: concatenated dataframes
    """
    dfs = [df.set_index(index_name) for df in list_of_dfs]
    df_result = pd.concat(dfs, axis=1)
    return df_result

def _add_suffix_except_col(df: pd.DataFrame, suffix: str, exception_col: str) -> pd.DataFrame:
    """Adds suffix to all columns except exception_col.

    Args:
        df (pd.DataFrame): dataframe
        suffix (str): suffix to add
        exception_col (str): exception_col

    Returns:
        pd.DataFrame: dataframe with changed column names
    """
    df.columns = [col+suffix if col!=exception_col else col for col in df.columns]
    return df

## ARTICLE FEATURES
def _rebuying_articles(transactions: pd.DataFrame) -> pd.DataFrame:
    """For each article_id calculate percentage of customers that bought this article_id more than once.

    Args:
        transactions (pd.DataFrame): transactions

    Returns:
        pd.DataFrame: dataframe with article_id and perc_rebought
    """
    df_rebought = (
        transactions
            .groupby(['customer_id', 'article_id'])['t_dat']
            .count()
            .reset_index()
            .assign(rebought=lambda x: (x.t_dat>1).astype(int))
            .groupby(['article_id'])['rebought']
            .mean()
            .reset_index(name='perc_rebought')
        )
    return df_rebought

def _mean_perc_sales_channel_id(transactions: pd.DataFrame) -> pd.DataFrame:
    """For each article_id calculate the share of customers that bought this article_id offline. 

    Args:
        transactions (pd.DataFrame): transactions

    Returns:
        pd.DataFrame: dataframe with article_id and perc_article_sales_offline
    """
    df_mean_sales_channel_id = (
        transactions
            # sales_channel_id=2 -> offline; sales_channel_id=1 -> online
            .assign(if_offline=lambda x: x.sales_channel_id-1)
            .groupby(['article_id'])['if_offline']
            .mean()
            .reset_index(name='perc_article_sales_offline')
    )
    return df_mean_sales_channel_id

def create_article_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Calculates all article features, then concatenates them into a single dataframe.

    Args:
        transactions (pd.DataFrame): transactions

    Returns:
        pd.DataFrame: article dataframe with multiple features
    """
    # features
    logger.info('Calculating article features...')
    df_rebought = _rebuying_articles(transactions)
    df_mean_sales_channel_id = _mean_perc_sales_channel_id(transactions)
    # concat
    logger.info('Concatenating article features...')
    df_result = _concat_dataframes_on_index([df_rebought, df_mean_sales_channel_id], 'article_id')
    return df_result

## CUSTOMER FEATURES
def _count_of_article_id_per_customer_id(transactions: pd.DataFrame, n_days: Optional[int]) -> pd.DataFrame:
    """Calculates count of article_id for each customer in the last n_days.

    Args:
        transactions (pd.DataFrame): transactions
        n_days (Optional[int]): number of days

    Returns:
        pd.DataFrame: dataframe with customer_id and count_of_article_per_customer
    """
    filtered_transactions = _filter_dataframe_by_last_n_days(transactions, n_days, date_column='t_dat')
    df_count_articles = (
        filtered_transactions
            .groupby(['customer_id'])['article_id']
            .count()
            .reset_index(name='count_of_article_per_customer')
    )
    return df_count_articles

def _count_of_product_group_name_per_customer_id(transactions: pd.DataFrame, articles: pd.DataFrame, n_days: Optional[int]) -> pd.DataFrame:
    """Calculates count of product_group_name for each customer in the last n_days.

    Args:
        transactions (pd.DataFrame): transactions
        articles (pd.DataFrame): articles
        n_days (Optional[int]): number of days

    Returns:
        pd.DataFrame: dataframe with customer_id and count_of_product_group_name_per_customer
    """
    filtered_transactions = _filter_dataframe_by_last_n_days(transactions, n_days, date_column='t_dat')
    df_count_pg = (
        filtered_transactions
            .merge(articles[['article_id', 'product_group_name']], on='article_id')[['customer_id', 'product_group_name']]
            .drop_duplicates()
            .groupby(['customer_id'])['product_group_name']
            .count()
            .reset_index(name='count_of_product_group_name_per_customer')
    )
    return df_count_pg

def _days_since_first_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Calculates number of days since first transaction for each customer.

    Args:
        transactions (pd.DataFrame): transactions

    Returns:
        pd.DataFrame: dataframe with customer_id and days_since_first_transaction
    """
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
    max_date = transactions['t_dat'].max()
    df_days_since_first = (
        max_date - (
            transactions
                .groupby(['customer_id'])['t_dat']
                .min()
        )
    ).dt.days.reset_index(name='days_since_first_transaction')
    return df_days_since_first

def _days_since_last_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Calculates number of days since last transaction for each customer.

    Args:
        transactions (pd.DataFrame): transactions

    Returns:
        pd.DataFrame: dataframe with customer_id and days_since_last_transaction
    """
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
    max_date = transactions['t_dat'].max()
    df_days_since_last = (
        max_date - (
            transactions
                .groupby(['customer_id'])['t_dat']
                .max()
        )
    ).dt.days.reset_index(name='days_since_last_transaction')
    return df_days_since_last

def _average_purchase_span(transactions: pd.DataFrame) -> pd.DataFrame:
    """Calculates average number of days between transactions for each customer.
    Customer must have at least 2 transactions.

    Args:
        transactions (pd.DataFrame): transactions

    Returns:
        pd.DataFrame: dataframe with customer_id and avg_purchase_span
    """
    sorted_transactions = (
        transactions
            .sort_values(by=['customer_id', 't_dat'], ascending=[True, False])[['customer_id', 't_dat']]
            .drop_duplicates()
    )
    sorted_transactions['t_dat_next'] = sorted_transactions['t_dat'].shift(1)
    sorted_transactions['customer_id_next'] = sorted_transactions['customer_id'].shift(1)
    sorted_transactions = sorted_transactions.assign(avg_purchase_span = lambda x: x.t_dat_next-x.t_dat)
    df_purchase_span = (
        sorted_transactions[sorted_transactions['customer_id']==sorted_transactions['customer_id_next']]
            .groupby(['customer_id'])['avg_purchase_span']
            .mean().dt.days
            .reset_index()
    )
    return df_purchase_span

def _perc_sales_channel_id(transactions: pd.DataFrame) -> pd.DataFrame:
    """Calculates percentage of offline sales for each customer.

    Args:
        transactions (pd.DataFrame): transactions

    Returns:
        pd.DataFrame: dataframe with customer_id and perc_customer_sales_offline
    """
    df_sales_channel_id = (
        transactions
            # sales_channel_id=2 -> offline; sales_channel_id=1 -> online
            .assign(if_offline=lambda x: x.sales_channel_id-1)
            .groupby(['customer_id'])['if_offline']
            .mean()
            .reset_index(name='perc_customer_sales_offline')
    )
    return df_sales_channel_id

def create_customer_features(transactions: pd.DataFrame, articles: pd.DataFrame, n_days_list: List[Optional[int]]) -> pd.DataFrame:
    """Calculates all customer features, then concatenates them into a single dataframe.

    Args:
        transactions (pd.DataFrame): transactions
        articles (pd.DataFrame): articles
        n_days_list (List[Optional[int]]): list of different time windows for some feature calculation
    
    Returns:
        pd.DataFrame: customer dataframe with multiple features
    """
    # different time windows features
    logger.info('Calculating time window customer features...')
    dfs_list = list()
    for n_days in n_days_list:
        suffix_str = '_all' if n_days is None else f'_{n_days}'
        df_count_articles = _count_of_article_id_per_customer_id(transactions, n_days)
        df_count_articles = _add_suffix_except_col(df_count_articles, suffix_str, 'customer_id')
        dfs_list.append(df_count_articles)
        df_count_pg = _count_of_product_group_name_per_customer_id(transactions, articles, n_days)
        df_count_pg = _add_suffix_except_col(df_count_pg, suffix_str, 'customer_id')
        dfs_list.append(df_count_pg)

    # another features
    logger.info('Calculating customer features...')
    df_days_since_first = _days_since_first_transactions(transactions)
    df_days_since_last = _days_since_last_transactions(transactions)
    df_purchase_span = _average_purchase_span(transactions)
    df_sales_channel_id = _perc_sales_channel_id(transactions)

    # concat
    logger.info('Concatenating customer features...')
    df_result = _concat_dataframes_on_index(
        [df_days_since_first, df_days_since_last, df_purchase_span, df_sales_channel_id, *dfs_list],
        'customer_id')
    return df_result

## CUSTOMER - product_group_name FEATURES
def _count_of_article_id_per_customer_product_group(transactions: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    """Calculates count of article_id for each customer and product_group_name.

    Args:
        transactions (pd.DataFrame): transactions
        articles (pd.DataFrame): articles

    Returns:
        pd.DataFrame: dataframe with article_id, product_group_name and count_of_article_per_customer_prod_group
    """
    df_count_article = (
        transactions
            .merge(articles[['article_id', 'product_group_name']], on='article_id')
            .groupby(['customer_id', 'product_group_name'])['article_id']
            .count()
            .reset_index(name='count_of_article_per_customer_prod_group')
    )
    return df_count_article

def create_customer_product_group_features(transactions: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    """Calculates all customer x product_group_name features, then concatenates them into a single dataframe.

    Args:
        transactions (pd.DataFrame): transactions
        articles (pd.DataFrame): articles

    Returns:
        pd.DataFrame: customer x product_group_name dataframe with multiple features
    """
    # features
    logger.info('Calculating customer x product_group_name features...')
    df_count_article = _count_of_article_id_per_customer_product_group(transactions, articles)
    # concat
    logger.info('Concatenating customer x product_group_name features...')
    df_result = _concat_dataframes_on_index([df_count_article], ['customer_id', 'product_group_name'])
    return df_result
