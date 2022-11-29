import pandas as pd
import pytest

from recommender_gnn.pipelines.graph_recommendation_preprocessing.nodes import (
    _create_mapping,
)


@pytest.fixture
def small_transactions_custom():
    transactions_dict = {
        "user_id": {
            "0": 0,
            "1": 1,
            "2": 1,
            "3": 2,
            "4": 3,
            "5": 4,
            "6": 5,
        },
        "item_id": {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 2,
            "4": 3,
            "5": 0,
            "6": 1,
        },
        "time": {
            "0": 1453939200,
            "1": 1453039200,
            "2": 1453032200,
            "3": 1453132200,
            "4": 1453132200,
            "5": 1453939201,
            "6": 1453939202,
        },
    }
    transactions_df = pd.DataFrame(transactions_dict)
    return transactions_df


@pytest.fixture
def small_predictions_custom():
    predictions_dict = {
        "0": {
            "0": 1,
            "1": 1,
            "2": 1,
            "3": 2,
            "4": 0,
            "5": 1,
        },
        "1": {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 3,
            "4": 1,
            "5": 0,
        },
        "2": {
            "0": 2,
            "1": 2,
            "2": 2,
            "3": 0,
            "4": 3,
            "5": 2,
        },
        "3": {
            "0": 3,
            "1": 3,
            "2": 3,
            "3": 1,
            "4": 2,
            "5": 3,
        },
        "user_id": {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
        },
    }
    predictions_df = pd.DataFrame(predictions_dict)
    return predictions_df


@pytest.fixture
def small_users_custom():
    users_dict = {
        "user_id": {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
        },
    }
    users_df = pd.DataFrame(users_dict)
    return users_df


@pytest.fixture
def small_users_mapping_custom(small_transactions_custom):
    map_column = "user_id"
    mapping = _create_mapping(small_transactions_custom, map_column)
    return mapping


@pytest.fixture
def small_items_mapping_custom(small_transactions_custom):
    map_column = "item_id"
    mapping = _create_mapping(small_transactions_custom, map_column)
    return mapping
