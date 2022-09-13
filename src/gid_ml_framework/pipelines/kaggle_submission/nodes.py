import logging
from typing import Callable, Dict, List, Tuple

import pandas as pd

from gid_ml_framework.extras.datasets.chunks_dataset import _concat_chunks

pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)


def _map_to_original(
    predictions: pd.DataFrame,
    user_mapping: Dict,
    item_mapping: Dict,
    user_column: str,
    item_columns: str,
) -> pd.DataFrame:
    """Maps mapped user and item ids to original ones"""
    reverse_user_mapping = dict((v, k) for k, v in user_mapping.items())
    reverse_item_mapping = dict((v, k) for k, v in item_mapping.items())
    predictions.loc[:, user_column] = predictions.loc[:, user_column].map(
        reverse_user_mapping.get
    )
    for item_column in item_columns:
        predictions.loc[:, item_column] = predictions.loc[:, item_column].map(
            reverse_item_mapping.get
        )
    return predictions


def _borda_sort(rankings: List):
    """Aggregates multiple rankings with the Borda count ranked voting system"""
    scores = {}
    for ranking in rankings:
        for idx, elem in enumerate(reversed(ranking)):
            if elem not in scores:
                scores[elem] = 0
            scores[elem] += idx
    return sorted(scores.keys(), key=lambda elem: scores[elem], reverse=True)


def _impute_missing_predictions(
    predictions: pd.DataFrame,
    user_mapping: Dict,
    user_column: str,
    item_columns: List[str],
    new_item_column: str,
) -> pd.DataFrame:
    """Generates prediction for users without any using Borda ranking aggregation method"""
    rankings = predictions.loc[:, item_columns].values.tolist()
    aggregated_ranking = _borda_sort(rankings)
    all_users = set(user_mapping.keys())
    pred_users = set(predictions.loc[:, [user_column]])
    left_users = list(all_users.difference(pred_users))
    left_predictions = pd.DataFrame({user_column: left_users})
    left_predictions.loc[:, new_item_column] = [aggregated_ranking] * len(
        left_predictions
    )
    return left_predictions


def _get_santander_columns(predictions: pd.DataFrame) -> Tuple:
    columns_list = set(predictions.columns)
    new_item_column = "added_products"
    user_column = "user_id"
    item_columns = list(columns_list.difference(set([user_column])))
    item_columns = sorted(item_columns, key=int)
    return user_column, item_columns, new_item_column


def generate_santander_submission(
    predictions: pd.DataFrame, user_mapping: Dict, item_mapping: Dict
) -> pd.DataFrame:
    """Generates santander submission file based on predictions from GNN model. It imputes predictions for missing
    users and remap user and item ids.

    Args:
        predictions (pd.DataFrame): dataframe with sorted predictions for each user with more than two transactions
        user_mapping (Dict): user mapping dict used to map original user ids to ones consistent with GNNs models
        item_mapping (Dict): user mapping dict used to map original user ids to ones consistent with GNNs models

    Returns:
        pd.DataFrame: dataframe with santander submission dataframe
    """
    predictions = _concat_chunks(predictions)
    user_column, item_columns, new_item_column = _get_santander_columns(predictions)
    predictions = _map_to_original(
        predictions, user_mapping, item_mapping, user_column, item_columns
    )
    left_predictions = _impute_missing_predictions(
        predictions, user_mapping, user_column, item_columns, new_item_column
    )
    predictions.loc[:, new_item_column] = predictions.loc[
        :, item_columns
    ].values.tolist()
    predictions = predictions.loc[:, [user_column, new_item_column]]
    submission = pd.concat([predictions, left_predictions])
    submission.rename(columns={user_column: "ncodpers"}, inplace=True)
    submission.loc[:, new_item_column] = submission.loc[:, new_item_column].str.join(
        " "
    )
    return submission


def generate_hm_submission():
    pass


def generate_submission(dataset: str) -> Callable:
    """Wrapper for dataset specific submission functions"""
    func_dict = {
        "santander": generate_santander_submission,
        "hm": generate_hm_submission,
    }
    return func_dict.get(dataset)
