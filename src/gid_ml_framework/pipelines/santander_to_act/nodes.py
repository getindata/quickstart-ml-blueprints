from typing import Iterator, Tuple
import logging
import re

import pandas as pd
import numpy as np
from kedro.extras.datasets.pandas import CSVDataSet

from gid_ml_framework.extras.datasets.chunks_dataset import (
 _load,
 _concat_chunks,
)


pd.options.mode.chained_assignment = None
log = logging.getLogger(__name__)
# Overwriting load method because of chunksize bug in Kedro < 0.18
CSVDataSet._load = _load


def santander_to_articles(santander: Iterator[pd.DataFrame]) \
    -> pd.DataFrame:
    """From Santander train/val dataset extract articles data

    Args:
        santander (pd.DataFrame): preprocessed train/val santander dataset

    Returns:
        pd.DataFrame: dataframe with features of each article
    """
    santander = _concat_chunks(santander)
    # Regex for Santander products names
    r = re.compile("ind_+.*ult.*")
    # There is no article features in Santander dataset 
    articles = pd.DataFrame({'article_id': list(filter(r.match,
                                                       santander.columns))})
    log.info(f"Number of unique articles: {articles.shape[0]}")
    log.info(f'Number of columns with missing values in articles dataset: \
    {articles.isnull().any().sum()}')                                             
    return articles



def santander_to_customers(santander: Iterator[pd.DataFrame],
                           merge_type: str) -> pd.DataFrame:
    """From Santander train/val/test dataset extract customers data

    Args:
        santander (pd.DataFrame): preprocessed train/val santander dataset
        merge_type (str): function which will be used to merge customer
            features from all timestamps (months) into one representation

    Returns:
        pd.DataFrame: dataframe with features of each customer
    """
    santander = _concat_chunks(santander)
    santander.rename(columns={'ncodpers': 'customer_id'}, inplace=True)
    if merge_type == 'last':
        # Customers features from last month
        last_month = santander.loc[:, 'fecha_dato'].max()
        customers = santander.loc[santander['fecha_dato'] >= last_month, :]
        customers.drop(['fecha_dato'], axis=1, inplace=True)
    else:
        # If no merge_type specified list of unique customer_id is returned
        customers = pd.DataFrame({'customer_id': santander['customer_id']
                                                 .unique()})
    log.info(f"Number of unique customers: {customers.shape[0]}")
    log.info(f"Number of customers features: {customers.shape[1]}")
    log.info(f'Number of columns with missing values in customers dataset: \
    {customers.isnull().any().sum()}')    
    return customers


def _status_change(x: pd.Series) -> str:
    """Based on difference of the following rows create label which indicates
    if given product was added, dropped or maintained in comparison with last 
    month for given customer.

    Args:
        x (pd.DataFrame): imputed santander train dataframe

    Returns:
        str: target label - added/dropped/maintained
    """
    # First occurrence is considered as "Maintained"
    label = ["Added" if i == 1 \
         else "Dropped" if i == -1 \
         else "Maintained" for i in x]
    return label


def _identify_newly_added(input_train_df: Iterator[pd.DataFrame],
                                input_val_df: Iterator[pd.DataFrame]) -> Tuple:
    """Preprocess target columns to identify products that will be bought
    in the next month

    Args:
        input_train_df (Iterator[pd.DataFrame]): imputed santander train
        dataframe
        input_val_df (Iterator[pd.DataFrame]): imputed santander validation 
        dataframe

    Returns:
        Tuple: processed train and validation dataframes
    """
    train_df = _concat_chunks(input_train_df)
    val_df = _concat_chunks(input_val_df)
    train_len = len(train_df)
    df = pd.concat([train_df, val_df])
    feature_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
    df = df.sort_values(['ncodpers', 'fecha_dato']).reset_index(drop=True)
    # Apply status change labeling
    s = df.loc[:, [i for i in feature_cols] + ["ncodpers"]]
    # Optimized version without groupby
    df.loc[:, feature_cols] = (s.loc[:, [i for i in feature_cols]].diff()
                               .where(s.duplicated(["ncodpers"],
                               keep='first'))
                               ).fillna(0).transform(_status_change)
    df = df.sort_values(['fecha_dato']).reset_index(drop=True)

    log.info(f'Sum of number of newly added products: \
             {df.eq("Added").sum().sum()}')

    train_df = df.iloc[:train_len, :]
    val_df = df.iloc[train_len:, :]
    return (train_df, val_df)


def _interaction_to_transaction(newly_added: pd.DataFrame, article_name: str) \
    -> pd.DataFrame:
    """Generate transactions records based on customer-article interactions
    for chosen article

    Args:
        newly_added (pd.DataFrame): santander dataframe in newly_added format
        article_name (str): name of chosen article columns

    Returns:
        pd.DataFrame: transactions for chosen article
    """
    article_interactions = newly_added.loc[:, ['ncodpers', 'fecha_dato',
                                              article_name]]
    # Filtering only newly added articles
    article_transactions = article_interactions.loc[article_interactions
                                                    .loc[:, article_name] ==
                                                    "Added", :]
    article_transactions['article_id'] = article_name
    article_transactions.drop(article_name, axis=1, inplace=True)
    article_transactions.rename(columns={'fecha_dato': 'date'}, inplace=True)
    return article_transactions


def _newly_added_to_transactions(newly_added: pd.DataFrame) -> pd.DataFrame:
    """Generate transactions dataframe based on dataframe in newly_added format

    Args:
        newly_added (pd.DataFrame): santander dataframe in newly_added format

    Returns:
        pd.DataFrame: transaction dataframe
    """
    # Regex for Santander products names
    r = re.compile("ind_+.*ult.*")
    articles_cols = list(filter(r.match, newly_added.columns))
    transactions = pd.DataFrame()
    # Generating transactions for each article
    for col in articles_cols:
        article_i_transactions = _interaction_to_transaction(newly_added, col)
        transactions = pd.concat(transactions, article_i_transactions,
                                 ignore_index=True)
    return transactions


def santander_to_transactions(santander_train: Iterator[pd.DataFrame],
                              santander_val: Iterator[pd.DataFrame]) \
    -> Tuple(pd.DataFrame, pd.DataFrame):
    """From Santander train/val datasets extract transactions data

    Args:
        santander_train (pd.DataFrame): preprocessed train santander dataframe
        santander_val (pd.DataFrame): preprocessed val santander dataframe

    Returns:
        Tuple(pd.DataFrame, pd.DataFrame): train and val transactions
            dataframes
    """
    newly_added_train, newly_added_val = _identify_newly_added(santander_train,
                                                               santander_val)
    # Apply for train and val dataframes
    transactions_train = _newly_added_to_transactions(newly_added_train)
    transactions_val = _newly_added_to_transactions(newly_added_val)
    return (transactions_train, transactions_val)