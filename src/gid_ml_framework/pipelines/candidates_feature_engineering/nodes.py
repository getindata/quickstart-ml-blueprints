import logging
import pandas as pd
from typing import Iterable, List


logger = logging.getLogger(__name__)

## JACCARD SIMILARITY
def _jaccard_similarity(x: Iterable, y: Iterable) -> float:
    """Returns the Jaccard similarity between two iterables.

    Args:
        x (Iterable): first iterable
        y (Iterable): second iterable

    Returns:
        float: Jaccard similarity between iterables x and y. Value should be between 0 and 1.
    """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

def _calculate_avg_jaccard_similarity(candidates_df: pd.DataFrame, articles_attributes: pd.DataFrame, transactions: pd.DataFrame) -> float:
    """Calculates the average Jaccard similarity between candidates and previous transactions.

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
        transactions (pd.DataFrame): transactions

    Returns:
        float: average Jaccard similarity
    """

    candidate_item, candidate_user = candidates_df.article_id, candidates_df.customer_id
    candidate_item_attributes = articles_attributes.loc[candidate_item]['set_of_attributes']
    bought_items = list(transactions[transactions['customer_id']==candidate_user].article_id.unique())
    if not bought_items:
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

def apply_avg_jaccard_similarity(candidates_df: pd.DataFrame, article_attributes: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Calculates average Jaccard similarity for all candidates, given transactions history.

    Args:
        candidates_df (pd.DataFrame): candidates (must have article_id, customer_id columns)
        article_attributes (pd.DataFrame): dataframe with article_id index and set_of_attributes column
        transactions (pd.DataFrame): transactions

    Returns:
    TODO: it may require change in the future
        pd.DataFrame: candidates dataframe with articles_jaccard_similarity
    """
    candidates_df['articles_jaccard_similarity'] = (
        candidates_df.apply(
            lambda x: _calculate_avg_jaccard_similarity(x, article_attributes, transactions),
            axis=1
            )
    )
    return candidates_df
