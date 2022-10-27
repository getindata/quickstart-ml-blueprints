import logging
import pandas as pd
import numpy as np
from typing import Set, List
from gid_ml_framework.helpers.utils import reduce_memory_usage


logger = logging.getLogger(__name__)

def filter_last_n_rows_per_customer(transactions: pd.DataFrame, last_n_rows: int) -> pd.DataFrame:
    """Filter transactions to latest n rows for each customer.

    Args:
        transactions (pd.DataFrame): transactions
        last_n_rows (int): number of latest rows for each customer

    Returns:
        pd.DataFrame: filtered transactions
    """
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
    sorted_transactions = transactions.sort_values(by='t_dat', ascending=False)
    latest_transactions = sorted_transactions.groupby(['customer_id']).head(last_n_rows)
    return latest_transactions

## CANDIDATES UNPACKING
def unpack_candidates(candidates_df: pd.DataFrame, drop_random_strategies: bool = False) -> pd.DataFrame:
    """Unpacks candidates and assigns strategy_name column with the name of generating candidates.
    If there are multiple candidates from more than one strategy, then duplicates are removed.

    Args:
        candidates_df (pd.DataFrame): candidates with list of articles as columns
        drop_random_strategies (bool): if true, random strategy will be used as feature; else - strategy_name='multiple_strategies'

    Returns:
        pd.DataFrame: long candidates
    """
    dfs = list()
    candidates_df.set_index('customer_id', inplace=True)
    for strategy_name in candidates_df.columns:
        logger.info(f'Unpacking {strategy_name}')
        df = (
            candidates_df[[strategy_name]]
                .explode(strategy_name)
                .rename({strategy_name: 'article_id'}, axis=1)
                .dropna()
        )
        df = df.assign(strategy_name=lambda x: strategy_name)
        logger.info(f'Finished unpacking {strategy_name} dataframe')
        dfs.append(df)
    logger.info(f'Concatenating candidate strategies into one long dataframe')
    long_candidates = pd.concat(dfs, axis=0).reset_index()
    # memory optimization
    long_candidates = reduce_memory_usage(long_candidates)
    logger.info(f'Long candidates df shape: {long_candidates.shape}')
    # removing duplicates (multiple strategies can have the same item)
    if drop_random_strategies:
        long_candidates = long_candidates.sample(frac=1).groupby(['customer_id', 'id_article']).head(1).reset_index()
        logger.info(f'Long candidates after dropping (random) multiple candidates: {long_candidates.shape}')
        return long_candidates
    long_candidates['count_strategy_name'] = (
        long_candidates
            .groupby(['customer_id', 'article_id'])['strategy_name']
            .transform('count')
    )
    long_candidates['strategy_name'] = np.where(
        long_candidates['count_strategy_name']>1,
        'multiple_strategies',
        long_candidates['strategy_name'])
    long_candidates = long_candidates.drop(['count_strategy_name'], axis=1).drop_duplicates()
    logger.info(f'Long candidates after dropping (not random) multiple candidates: {long_candidates.shape}')
    return long_candidates

## JACCARD SIMILARITY
def _jaccard_similarity(x: Set, y: Set) -> float:
    """Returns the Jaccard similarity between two sets.

    Args:
        x (Iterable): first set
        y (Iterable): second set

    Returns:
        float: Jaccard similarity between sets x and y. Value should be between 0 and 1.
    """
    intersection_cardinality = len(x.intersection(y))
    union_cardinality = len(x.union(y))
    return float(intersection_cardinality/union_cardinality)

def _calculate_avg_jaccard_similarity(candidates_df: pd.DataFrame, articles_attributes: pd.DataFrame, customer_list_of_articles: pd.DataFrame) -> float:
    """Calculates the average Jaccard similarity between candidates and previous transactions. If not found, returns 0.

    So when a customer had 3 articles in the transactions history:
        item_1 attributes {A, B, C}
        item_2 attributes {B, C, D}
        item_3 attributes {A, B, D}

    Then you consider a new item_4 with attributes {A, B}, the average Jaccard similarity is calculated like this:
        item_1 attributes {A, B, C} -> Jaccard = 0.66  
        item_2 attributes {B, C, D} -> Jaccard = 0.33  
        item_3 attributes {A, B, D} -> Jaccard = 0.66

    So, the final average Jaccard similarity is (0.66+0.33+0.66)/3 = 0.55

    Args:
        candidates_df (pd.DataFrame): candidates (article_id, customer_id)
        articles_attributes (pd.DataFrame): dataframe with article_id as index, and set_of_attributes as set of attributes
        customer_list_of_articles (pd.DataFrame): dataframe with customer_id and list of articles previously bought

    Returns:
        float: average Jaccard similarity
    """
    candidate_item, candidate_user = candidates_df.article_id, candidates_df.customer_id
    candidate_item_attributes = articles_attributes.loc[candidate_item]['set_of_attributes']
    try:
        bought_items = customer_list_of_articles.loc[candidate_user][0]
    except KeyError:
        return 0
    jaccard_similarity_list = []
    for item in bought_items:
        item_attributes = articles_attributes.loc[item]['set_of_attributes']
        jaccard_similarity = _jaccard_similarity(candidate_item_attributes, item_attributes)
        jaccard_similarity_list.append(jaccard_similarity)
    return sum(jaccard_similarity_list)/len(jaccard_similarity_list)

def create_set_of_attributes(articles: pd.DataFrame, attribute_cols: List) -> pd.DataFrame:
    """Creates a set of attributes column from attribute_cols list.

    Args:
        articles (pd.DataFrame): articles
        attribute_cols (List): list of attributes

    Returns:
        pd.DataFrame: dataframe with article_id index and set_of_attributes column
    """
    cat_articles = articles[attribute_cols]
    articles_attributes = articles[['article_id']].copy()
    articles_attributes['set_of_attributes'] = cat_articles.apply(set, axis=1)
    articles_attributes.set_index(['article_id'], inplace=True)
    return articles_attributes

def create_list_of_previously_bought_articles(transactions: pd.DataFrame) -> pd.DataFrame:
    """Creates a dataframe with customer_id and list of previously bought articles.

    Args:
        transactions (pd.DataFrame): transactions

    Returns:
        pd.DataFrame: dataframe with customer_id and list_of_articles column
    """
    customer_list_of_articles = (
        transactions[['customer_id', 'article_id']]
            .drop_duplicates()
            .groupby(['customer_id'])['article_id']
            .apply(list)
            .reset_index(name='list_of_articles')
            .set_index('customer_id')
    )
    return customer_list_of_articles

def apply_avg_jaccard_similarity(candidates_df: pd.DataFrame, article_attributes: pd.DataFrame, customer_list_of_articles: pd.DataFrame) -> pd.DataFrame:
    """Calculates average Jaccard similarity for all candidates, given transaction history.

    Args:
        candidates_df (pd.DataFrame): candidates (must have article_id, customer_id columns)
        article_attributes (pd.DataFrame): dataframe with article_id index and set_of_attributes column
        customer_list_of_articles (pd.DataFrame): dataframe with customer_id and list of articles previously bought

    Returns:
        pd.DataFrame: candidates dataframe with articles_jaccard_similarity
    """
    logger.info(f'Applying average jaccard similarity.')
    candidates_df['articles_jaccard_similarity'] = (
        candidates_df.apply(
            lambda x: _calculate_avg_jaccard_similarity(x, article_attributes, customer_list_of_articles),
            axis=1
            )
    )
    return candidates_df

## COSINE SIMILARITY
def _mean_customer_embeddings(customer_list_of_articles: pd.DataFrame, embeddings: pd.DataFrame) -> pd.DataFrame:
    """Calculates embeddings for a single customer as a mean of his previous article purchases.

    Args:
        customer_list_of_articles (pd.DataFrame): dataframe with customer_id and list of articles previously bought
        embeddings (pd.DataFrame): embeddings

    Returns:
        pd.DataFrame: mean embeddings
    """
    list_of_articles = customer_list_of_articles.list_of_articles[0]
    mean_embeddings = list(embeddings[embeddings.index.isin(list_of_articles)].mean(axis=0))
    return mean_embeddings

def calculate_customer_embeddings(customer_list_of_articles: pd.DataFrame, embeddings: pd.DataFrame) -> pd.DataFrame:
    """For each customer calculates mean embeddings.

    Args:
        customer_list_of_articles (pd.DataFrame): dataframe with customer_id and list of articles previously bought
        embeddings (pd.DataFrame): embeddings

    Returns:
        pd.DataFrame: customer_id, embeddings dataframe
    """
    
    customer_embeddings = (
        customer_list_of_articles
            .groupby(['customer_id'])
            .apply(lambda x: _mean_customer_embeddings(x, embeddings))
            .reset_index(name='embeddings')
            .set_index('customer_id')
    )
    return customer_embeddings

def _cosine_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Calculates cosine similarity between 2 vectors.

    Args:
        A (np.ndarray): vector with numerical values
        B (np.ndarray): vector with numerical values

    Returns:
        float: cosine similarity. Should be between -1 and 1.
    """
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def _cosine_embedding_similarity(candidates_df: pd.DataFrame, customers_embeddings: pd.DataFrame, articles_embeddings: pd.DataFrame) -> pd.DataFrame:
    """Calculates cosine similarity for a single candidate item. If not found, returns 0.

    Args:
        candidates_df (pd.DataFrame): candidates
        customers_embeddings (pd.DataFrame): customer embeddings
        articles_embeddings (pd.DataFrame): articles ambeddings

    Returns:
        float: cosine similarity
    """
    candidate_item, candidate_user = candidates_df.article_id, candidates_df.customer_id
    try:
        customer_embeddings = np.array(customers_embeddings.loc[candidate_user].embeddings)
    # no transactions
    except KeyError:
        return 0
    try:
        candidate_embeddings = articles_embeddings.loc[candidate_item].values
    # no image/no text for given article
    except KeyError:
        return 0
    return _cosine_similarity(customer_embeddings, candidate_embeddings)

def apply_cosine_similarity(candidates_df: pd.DataFrame, customer_embeddings: pd.DataFrame, article_embeddings: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Calculates cosine similarity for all candidates, given transaction history and embeddings.

    Args:
        candidates_df (pd.DataFrame): candidates
        customer_embeddings (pd.DataFrame): customer embeddings
        article_embeddings (pd.DataFrame): articles embeddings
        col_name (str): new column name with cosine similarity

    Returns:
        pd.DataFrame: candidates dataframe with f'{col_name}_cosine_similarity'
    """
    col_name = f'{col_name}_cosine_similarity'
    logger.info(f'Applying cosine embedding similarity for {col_name}')
    candidates_df[col_name] = (
        candidates_df.apply(
            lambda x: _cosine_embedding_similarity(x, customer_embeddings, article_embeddings),
            axis=1)
    )
    return candidates_df

## MERGE SIMILARITIES
def merge_similarity_features(jaccard_candidates: pd.DataFrame, image_cosine_candidates: pd.DataFrame, text_cosine_candidates: pd.DataFrame) -> pd.DataFrame:
    """Merges multiple candidate dataframes with different similarities.

    Args:
        jaccard_candidates (pd.DataFrame): candidates dataframe with Jaccard similarity
        image_cosine_candidates (pd.DataFrame): candidates dataframe with image cosine similarity
        text_cosine_candidates (pd.DataFrame): candidates dataframe with text cosine similarity

    Returns:
        pd.DataFrame: candidates dataframe with Jaccard, image & text cosine similarity
    """
    logger.info(f'''Merging all features:
    Jaccard features: {jaccard_candidates.shape}
    image cosine features: {image_cosine_candidates.shape}
    text cosine features: {text_cosine_candidates.shape}''')
    merge_cols = ['customer_id', 'article_id', 'strategy_name']
    candidates_df = jaccard_candidates.merge(image_cosine_candidates, on=merge_cols).merge(text_cosine_candidates, on=merge_cols)
    logger.info(f'Merged features shape: {candidates_df.shape}')
    return candidates_df
