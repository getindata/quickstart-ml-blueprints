import pandas as pd
import numpy as np
import logging
from typing import List, Set
import mlflow
from gid_ml_framework.helpers.utils import reduce_memory_usage


log = logging.getLogger(__name__)

def _get_candidate_column_names(candidates: pd.DataFrame) -> List:
    """Returns all column names for different candidate generation methods.

    Args:
        candidates (pd.DataFrame): candidates

    Returns:
        List: different candidate generation methods
    """
    col_list = [col for col in list(candidates.columns) if col != 'customer_id']
    return col_list

def _get_unique_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Removes repeated purchases from transactions

    Args:
        transactions (pd.DataFrame): transactions

    Returns:
        pd.DataFrame: filtered transactions
    """
    log.info(f'Transactions shape before removing duplicates (repeated purchases): {transactions.shape}')
    transactions = transactions[['customer_id', 'article_id']].drop_duplicates()
    log.info(f'Transactions shape after removing duplicates: {transactions.shape}')
    return transactions

def _get_recall(candidates: pd.DataFrame, val_transactions: pd.DataFrame, candidate_col: str) -> float:
    """Calculates recall for given candidate generation method. Candidate method column should have 
    list-like data.

    Args:
        candidates (pd.DataFrame): candidates
        val_transactions (pd.DataFrame): validation transactions (unseen) on which to calculate recall
        candidate_col (str): column name pointing to candidate generation method

    Returns:
        float: recall, should be between 0 and 1
    """
    exploded_candidates_df = candidates[['customer_id', candidate_col]].explode(candidate_col)
    exploded_candidates_df['recall'] = 1
    # memory optimization
    exploded_candidates_df = reduce_memory_usage(exploded_candidates_df)
    val_transactions = val_transactions.merge(
        exploded_candidates_df,
        left_on=['customer_id', 'article_id'],
        right_on=['customer_id', candidate_col],
        how='left'
    )
    recall = val_transactions['recall'].fillna(0).mean()
    return recall

def _concatenate_all_lists(row: np.ndarray, no_cols: int) -> Set:
    """Helper function to `np.apply_along_axis`.
    Concatenates all candidates from multiple methods. It returns set to keep only unique values.

    Args:
        row (np.ndarray): row, which can be indexed
        no_cols (int): number of cols to aggregate

    Returns:
        Set: unique items from all columns
    """
    articles_set = set()
    for i in range(no_cols):
        if row[i] is None:
            continue
        articles_set |= set(row[i])
    return articles_set

def log_retrieval_recall(candidates: pd.DataFrame, val_transactions: pd.DataFrame) -> None:
    """Logs recall to MLflow for all candidate generation methods and jointly.

    Args:
        candidates (pd.DataFrame): candidates
        val_transactions (pd.DataFrame): validation transactions
    """
    candidates_cols_list = _get_candidate_column_names(candidates)
    val_transactions = _get_unique_transactions(val_transactions)
    # all candidates
    candidates['all_candidates'] = np.apply_along_axis(
        _concatenate_all_lists,
        axis=1,
        arr=candidates[candidates_cols_list].values,
        no_cols=len(candidates_cols_list)
    )
    candidates['all_candidates'] = candidates['all_candidates'].apply(list)
    candidates_cols_list = candidates_cols_list + ['all_candidates']
    for candidate_col in candidates_cols_list:
        log.info(f'Calculating recall for {candidate_col}')
        recall = _get_recall(candidates, val_transactions, candidate_col)
        mlflow.log_metric(f'{candidate_col}_recall', recall)
        # memory optimization
        candidates = candidates.drop(candidate_col, axis=1)
    