import logging
from typing import Dict, Iterator, List, Tuple, Union

import pandas as pd

from recommender_gnn.extras.datasets.chunks_dataset import _concat_chunks

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
    all_users: pd.DataFrame,
    user_column: str,
    item_columns: List[str],
    new_item_column: str,
    original_user_column: str,
) -> pd.DataFrame:
    """Generates prediction for users without any using Borda ranking aggregation method"""
    rankings = predictions.loc[:, item_columns].values.tolist()
    aggregated_ranking = _borda_sort(rankings)
    all_users = set(all_users.loc[:, original_user_column])
    pred_users = set(predictions.loc[:, user_column])
    left_users = list(all_users.difference(pred_users))
    left_predictions = pd.DataFrame({user_column: left_users})
    left_predictions.loc[:, new_item_column] = [aggregated_ranking] * len(
        left_predictions
    )
    return left_predictions


def _get_columns(predictions: pd.DataFrame) -> Tuple:
    """Returns list of user and items columns names needed for further processing"""
    columns_list = set(predictions.columns)
    user_column = "user_id"
    item_columns = list(columns_list.difference(set([user_column])))
    item_columns = sorted(item_columns, key=int)
    return user_column, item_columns


def _filter_by_test_users(
    submission: pd.DataFrame,
    test_input: Union[Iterator[pd.DataFrame], pd.DataFrame],
    test_user_column: str,
):
    """Filters submission dataframe with users ids present in test file"""
    test_input = _concat_chunks(test_input)
    test_users = set(test_input.loc[:, test_user_column])
    submission = submission.loc[submission[test_user_column].isin(test_users), :]
    return submission


def generate_submission(
    predictions: pd.DataFrame,
    all_users: pd.DataFrame,
    user_mapping: Dict,
    item_mapping: Dict,
    test_input: Union[Iterator[pd.DataFrame], pd.DataFrame],
    new_item_column: str,
    new_user_column: str,
    original_user_column: str,
    filter_by_test_users: bool,
) -> pd.DataFrame:
    """Generates kaggle submission file based on predictions from model. It imputes predictions for missing
    users, remap user and item ids and apply required formatting.

    Args:
        predictions (pd.DataFrame): dataframe with sorted predictions for each user with more than two transactions
        all_users (pd.DataFrame): dataframe with all users data
        user_mapping (Dict): user mapping dict used to map original user ids to ones consistent with GNNs models
        item_mapping (Dict): user mapping dict used to map original user ids to ones consistent with GNNs models
        test_input (pd.DataFrame): dataframe containing user ids from test subset
        new_item_column (str): name of item column required by Kaggle submission format
        new_user_column (str): name of user column required by Kaggle submission format

    Returns:
        pd.DataFrame: dataframe with kaggle submission dataframe
    """
    predictions = _concat_chunks(predictions)
    user_column, item_columns = _get_columns(predictions)
    predictions = _map_to_original(
        predictions,
        user_mapping,
        item_mapping,
        user_column,
        item_columns,
    )
    left_predictions = _impute_missing_predictions(
        predictions,
        all_users,
        user_column,
        item_columns,
        new_item_column,
        original_user_column,
    )
    predictions.loc[:, new_item_column] = predictions.loc[
        :, item_columns
    ].values.tolist()
    predictions = predictions.loc[:, [user_column, new_item_column]]
    submission = pd.concat([predictions, left_predictions], ignore_index=True)
    submission.rename(columns={user_column: new_user_column}, inplace=True)
    submission.loc[:, new_item_column] = submission.loc[:, new_item_column].str.join(
        " "
    )
    if filter_by_test_users:
        submission = _filter_by_test_users(submission, test_input, new_user_column)
    return submission
