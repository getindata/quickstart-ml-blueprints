import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union
from pathlib import Path
import shutil
from datetime import datetime
from typing import Union, Iterable
import logging
from sklearn.model_selection import train_test_split


log = logging.getLogger(__name__)

def filter_out_old_transactions(transactions: pd.DataFrame, cutoff_date: Union[str, datetime]='2020-04-01') -> pd.DataFrame:
    """Filter out the transactions by cutoff_date.

    Args:
        transactions: raw data
    Returns:
        pd.DataFrame: latest transactions
    """
    log.info(f"Transactions' shape before cutoff: {transactions.shape}")
    transactions.t_dat = pd.to_datetime(transactions.t_dat)
    transactions = transactions[transactions.t_dat>=cutoff_date]
    log.info(f"Transactions' shape after cutoff: {transactions.shape}")
    return transactions

def _copy_images(img_src_dir: Path, img_dst_dir: Path, article_ids: Iterable) -> None:
    """Copies filtered images from source directory to destination directory.

    Args:
        img_src_dir (Path): path to source directory with images
        img_dst_dir (Path): path to destination directory with images
        article_ids (Iterable): any iterable which consists of article_ids for filtering
    """
    img_files = [img_file for img_file in img_src_dir.glob('*.jpg') if img_file.name.split('.')[0] in article_ids]
    log.info(f"Number of image files to be copied: {len(img_files)}")
    Path.mkdir(img_dst_dir, exist_ok=True)
    for file in img_files:
        dst_file = img_dst_dir / file.name
        shutil.copy(file, dst_file)
    log.info(f"There are {sum(1 for _ in img_dst_dir.glob('*.jpg'))} files in the destination folder.")

def sample_articles(articles: pd.DataFrame, article_img_dir: str, article_img_sample_dir: str, sample_size: float=0.2) -> pd.DataFrame:
    """Sample articles based on sample_size. Copy ONLY SAMPLE article images from article_img_dir to article_img_sample_dir.

    Args:
        articles (pd.DataFrame): raw data
        articles_dir (str): path to source directory with images, name of an image must be consistent with article_id
        articles_new_dir (str): path to destination directory with images, only sampled articles are copied
        sample_size (float, optional): sample fraction. Defaults to 0.2.

    Returns:
        pd.DataFrame: sampled articles
    """
    log.info(f"Articles' shape before sampling {articles.shape}")
    # sampling articles dictionary
    articles_sample = articles.sample(frac=sample_size, random_state=321)
    log.info(f"Articles' shape before sampling {articles_sample.shape}")
    
    # sampling images
    article_ids = set(articles_sample.article_id.unique())
    img_src_dir = Path.cwd() / article_img_dir
    img_dst_dir = Path.cwd() / article_img_sample_dir
    _copy_images(img_src_dir, img_dst_dir, article_ids)
    return articles_sample

def _calculate_distinct_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create aggregate dataframe with number of unique days per customer.

    Args:
        transactions: raw data
    Returns:
        pd.DataFrame: transformed dataframe, with customer_id and number of unique days of transactions
    """
    customers_trans_cd = transactions.groupby(['customer_id'])['t_dat'].nunique().reset_index(name='no_transactions')
    return customers_trans_cd

def sample_customers(customers: pd.DataFrame, transactions: pd.DataFrame, sample_size: float=0.1) -> pd.DataFrame:
    """Sample customers based on number of unique days with transactions and age.

    Args:
        customers: raw data
        transactions: raw data
        sample_size: sample fraction

    Returns:
        pd.DataFrame: sampled customers
    """
    if np.isclose(sample_size, 1.0):
        customers_sample = customers
        log.info(f"Sample size for customer is {sample_size=}, there will be no sampling!")
        return customers_sample
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

def sample_transactions(transactions: pd.DataFrame, customers_sample: pd.DataFrame, articles_sample: pd.DataFrame) -> pd.DataFrame:
    """Filters out transactions based on sampled customers and articles pd.DataFrame's

    Args:
        transactions (pd.DataFrame): latest transactions raw data
        customers_sample (pd.DataFrame): sampled customers
        articles_sample (pd.DataFrame): sampled articles

    Returns:
        pd.DataFrame: sampled transactions
    """
    log.info(f"Transactions' shape before sampling: {transactions.shape}")
    unique_customer_id_set = set(customers_sample['customer_id'].unique())
    unique_article_ids_set = set(articles_sample['article_id'].unique())
    transactions_sample = transactions[(transactions['customer_id'].isin(unique_customer_id_set)) & (transactions['article_id'].isin(unique_article_ids_set))]
    log.info(f"Transactions' shape after sampling: {transactions_sample.shape}")
    return transactions_sample
