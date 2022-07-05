import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union
import logging
from sklearn.model_selection import train_test_split


log = logging.getLogger(__name__)

def filter_out_old_transactions(transactions: pd.DataFrame, cutoff_date: Union[str, datetime]='2020-04-01') -> pd.DataFrame:
    """Filter out the transactions by cutoff_date.

    Args:
        transactions: raw data
    Returns:
        Preprocessed dataframe, with only the latest transactions
    """
    log.info(f"Transactions' shape before cutoff: {transactions.shape}")
    transactions.t_dat = pd.to_datetime(transactions.t_dat)
    transactions = transactions[transactions.t_dat>=cutoff_date]
    log.info(f"Transactions' shape after cutoff: {transactions.shape}")
    return transactions

def _calculate_distinct_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create aggregate dataframe with number of unique days per customer.

    Args:
        transactions: raw data
    Returns:
        Transformed dataframe, with customer_id and number of unique days of transactions
    """
    customers_trans_cd = transactions.groupby(['customer_id'])['t_dat'].nunique().reset_index(name='no_transactions')
    return customers_trans_cd

def sample_customers(customers: pd.DataFrame, transactions: pd.DataFrame, sample_size: float=0.1) -> pd.DataFrame:
    """
    Sample customers based on number of unique days with transactions and age.

    Args:
        customers: raw data
        transactions: raw data
        SAMPLE_SIZE: sample fraction

    Returns:
        Sampled customers' dataframe
    """
    log.info(f"Customers' shape before sampling: {customers.shape}")
    customers_trans_cd = _calculate_distinct_transactions(transactions)
    customers = customers.merge(customers_trans_cd, on='customer_id', how='left')
    customers['age'] = customers['age'].fillna(customers['age'].median())
    customers['age_bin'] = pd.qcut(customers['age'], q=4, labels=['1', '2', '3', '4']).astype(str)
    customers['no_transactions'] = customers['no_transactions'].fillna(0)
    cond_transactions_list = [
        customers['no_transactions']==0,
        customers['no_transactions']<=1,
        customers['no_transactions']<=3,
        customers['no_transactions']<=6,
        customers['no_transactions']<=9
    ]
    choice_transactions_list = [
        '1', '2', '3', '4', '5'
    ]
    customers['no_transactions_bin'] = np.select(cond_transactions_list, choice_transactions_list, default='6')
    _, customers_sample = train_test_split(
        customers,
        test_size=sample_size, 
        stratify=customers[['age_bin', 'no_transactions_bin']]
    )
    customers_sample.drop(['no_transactions', 'age_bin', 'no_transactions_bin'], axis=1, inplace=True)
    log.info(f"Customers' shape after sampling: {customers_sample.shape}")
    return customers_sample

def sample_transactions(transactions: pd.DataFrame, customers_sample: pd.DataFrame) -> pd.DataFrame:
    log.info(f"Transactions' shape before sampling: {transactions.shape}")
    unique_customer_id_set = list(customers_sample['customer_id'].unique())
    transactions_sample = transactions[transactions['customer_id'].isin(unique_customer_id_set)]
    log.info(f"Transactions' shape after sampling: {transactions_sample.shape}")
    return transactions_sample
