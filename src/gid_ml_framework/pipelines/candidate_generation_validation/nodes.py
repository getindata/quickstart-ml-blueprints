import pandas as pd
import logging
from typing import List
import mlflow


log = logging.getLogger(__name__)

def _get_candidate_column_names(candidates: pd.DataFrame) -> List:
    col_list = [col for col in list(candidates.columns) if col != 'customer_id']
    return col_list

def _get_unique_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    log.info(f'Transactions shape before removing duplicates (repeated purchases): {transactions.shape}')
    transactions = transactions[['customer_id', 'article_id']].drop_duplicates()
    log.info(f'Transactions shape after removing duplicates: {transactions.shape}')
    return transactions

def _get_recall(candidates: pd.DataFrame, val_transactions: pd.DataFrame, candidate_col: str) -> float:
    exploded_candidates_df = candidates[['customer_id', candidate_col]].explode(candidate_col)
    exploded_candidates_df['recall'] = 1
    val_transactions = val_transactions.merge(
        exploded_candidates_df,
        left_on=['customer_id', 'article_id'],
        right_on=['customer_id', candidate_col],
        how='left'
    )
    recall = val_transactions['recall'].fillna(0).mean()
    return recall

def _concatenate_all_lists(*args) -> List:
    articles_set = set()
    for arg in args:
        if arg is None:
            continue
        new_items = set(arg)
        articles_set |= new_items
    return list(articles_set)

def log_retrieval_recall(candidates: pd.DataFrame, val_transactions: pd.DataFrame) -> None:
    candidates_cols_list = _get_candidate_column_names(candidates)
    val_transactions = _get_unique_transactions(val_transactions)
    for candidate_col in candidates_cols_list:
        log.info(f'Calculating recall for {candidate_col}')
        recall = _get_recall(candidates, val_transactions, candidate_col)
        mlflow.log_metric(f'{candidate_col}_recall', recall)
    # log recall for all candidates
    # HARD CODED - dunno yet how to make it general
    log.info(f'Calculating recall for all candidates')
    candidates['all_candidates'] = candidates.apply(lambda x: _concatenate_all_lists(x.global_articles,
                                             x.segment_articles,
                                             x.previously_bought,
                                             x.previously_bought_prod_name,
                                             x.closest_image_embeddings,
                                             x.closest_text_embeddings
                                            ), axis=1)
    recall = _get_recall(candidates, val_transactions, 'all_candidates')
    mlflow.log_metric(f'all_candidates_recall', recall)
    