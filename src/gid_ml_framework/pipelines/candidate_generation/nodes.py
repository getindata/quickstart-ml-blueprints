import pandas as pd
from typing import Union, Set, List, Dict
import logging


log = logging.getLogger(__name__)
ArticlesSet = Set
Bin = str
ArticlesBin = Dict[Bin, ArticlesSet]

# GLOBAL (for all users) ARTICLES
def _filter_dataframe_by_last_n_days(df: pd.DataFrame, n_days: int, date_column: str) -> pd.DataFrame:
    if not n_days:
        log.info(f'n_days is equal to None, skipping the filtering by date step.')
        return df
    df.loc[:, date_column] = pd.to_datetime(df.loc[:, date_column])
    max_date = df.loc[:, date_column].max()
    filter_date = max_date - pd.Timedelta(days=n_days)
    log.info(f'Maximum date is: {max_date}, date for filtering is: {filter_date}, {n_days=}')
    log.info(f'Shape before filtering: {df.shape}')
    df = df[df.loc[:, date_column]>=filter_date]
    log.info(f'Shape after filtering: {df.shape}')
    return df

def _most_sold_articles(transactions: pd.DataFrame, top_k: int) -> ArticlesSet:
    articles_sold_most = (transactions
        .groupby(['article_id'])['customer_id']
        .size()
        .reset_index(name='sales_sum_count')
        .sort_values(by='sales_sum_count', ascending=False)
        .head(top_k)['article_id']
    )
    articles_set = set(articles_sold_most)
    return articles_set

def collect_global_articles(transactions: pd.DataFrame, n_days_list: List[Union[int, None]], top_k_list: List[int]) -> ArticlesSet:
    all_global_articles = set()
    for n_days, top_k in zip(n_days_list, top_k_list):
        latest_transactions = _filter_dataframe_by_last_n_days(transactions, n_days, date_column='t_dat')
        articles_set = _most_sold_articles(latest_transactions, top_k)
        log.info(f'All articles size: {len(all_global_articles)} before adding articles from {n_days=}, {top_k=}')
        all_global_articles.update(articles_set)
        log.info(f'All articles size after: {len(all_global_articles)}')
    return all_global_articles

def global_articles(customers: pd.DataFrame, articles: ArticlesSet) -> pd.DataFrame:
    all_customers = customers[['customer_id']].copy()
    all_customers.loc[:, 'global_articles'] = pd.Series(list(articles) * len(all_customers))
    return all_customers

# SEGMENT ARTICLES
def segment_by_customer_age(customers: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    customers['segment_bin'] = pd.qcut(customers['age'], q=n_bins, labels=False).astype(str).str.zfill(2)
    return customers[['customer_id', 'segment_bin']]

def _filter_transactions_by_bin(transactions: pd.DataFrame, customers: pd.DataFrame, bin: str) -> pd.DataFrame:
    filtered_customers = customers[customers['segment_bin']==bin]
    filtered_transactions = transactions.merge(filtered_customers[['customer_id']], on='customer_id')
    return filtered_transactions

def _most_sold_articles_by_segment(transactions: pd.DataFrame, customers_bins: pd.DataFrame, top_k: int) -> ArticlesBin:
    articles_per_bin_dict = dict()
    unique_bins = list(customers_bins['segment_bin'].unique())
    for bin in unique_bins:
        bin_transactions = _filter_transactions_by_bin(transactions, customers_bins, bin)
        articles_set = _most_sold_articles(bin_transactions, top_k)
        articles_per_bin_dict[bin] = articles_set
    return articles_per_bin_dict

def _update_dict_of_sets(cumulative_dict: Dict, new_dict: Dict) -> Dict:
    for k, v in new_dict.items():
        cumulative_dict[k] = cumulative_dict.get(k, set()) | v
        log.info(f'Bin: {k=}, size of set: {len(v)=}, size of acc. set: {len(cumulative_dict[k])=}')
    return cumulative_dict

def collect_segment_articles(transactions: pd.DataFrame, customers_bins: pd.DataFrame, n_days_list: List[Union[int, None]], top_k_list: List[int]) -> ArticlesBin:
    segment_articles = dict()
    for n_days, top_k in zip(n_days_list, top_k_list):
        latest_transactions = _filter_dataframe_by_last_n_days(transactions, n_days, date_column='t_dat')
        articles_per_bin_dict = _most_sold_articles_by_segment(latest_transactions, customers_bins, top_k)
        segment_articles = _update_dict_of_sets(segment_articles, articles_per_bin_dict)
    return segment_articles

def segment_articles(articles_dict: ArticlesBin, customers_segment: pd.DataFrame) -> pd.DataFrame:
    articles_df = pd.DataFrame(articles_dict.keys(), columns=['segment_bin'])
    articles_df['segment_articles'] = articles_df.loc[:, 'segment_bin'].map(articles_dict)
    segment_articles = (
        customers_segment
        .merge(articles_df, on='segment_bin')
        .drop(['segment_bin'], axis=1)
    )
    return segment_articles

# PREVIOUSLY BOUGHT ARTICLES
def previously_bought_articles(transactions: pd.DataFrame) -> pd.DataFrame:
    prev_bought = transactions[['customer_id', 'article_id']].drop_duplicates()
    prev_bought = prev_bought.groupby(['customer_id'])['article_id'].apply(list).reset_index(name='previously_bought')
    return prev_bought

def previously_bought_prod_name_articles(transactions: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    articles_prod_name = articles[['article_id', 'prod_name']]
    customers_prod_name = (
        transactions
        .merge(articles_prod_name, on='article_id')[['customer_id', 'prod_name']]
        .drop_duplicates()
    )
    prev_bought_prod_name = (
        customers_prod_name
        .merge(articles_prod_name, on='prod_name')
        .drop(['prod_name'], axis=1)
    )
    prev_bought_prod_name = prev_bought_prod_name.groupby(['customer_id'])['article_id'].apply(list).reset_index(name='previously_bought_prod_name')
    return prev_bought_prod_name

# COLLECT ALL
def collect_all_candidates(global_articles: pd.DataFrame, segment_articles: pd.DataFrame,
                prev_bought_articles: pd.DataFrame, prev_bough_prod_name: pd.DataFrame) -> pd.DataFrame:
    collected_df = (
        global_articles
        .merge(segment_articles, on='customer_id', how='left')
        .merge(prev_bought_articles, on='customer_id', how='left')
        .merge(prev_bough_prod_name, on='customer_id', how='left')
        )
    return collected_df
