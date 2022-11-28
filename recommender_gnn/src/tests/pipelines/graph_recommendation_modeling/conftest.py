import pickle

import pandas as pd
import pytest

from recommender_gnn.pipelines.graph_recommendation_modeling.nodes import (
    generate_graph_dgsr,
)


@pytest.fixture
def mapped_transactions():
    path = "src/tests/fixtures/dataframes/santander_mapped_transactions.csv"
    mapped_df = pd.read_csv(path)
    return mapped_df


@pytest.fixture
def mapped_transactions_small():
    path = "src/tests/fixtures/dataframes/santander_mapped_transactions_small.csv"
    mapped_df = pd.read_csv(path)
    return mapped_df


@pytest.fixture
def mapped_transactions_custom():
    transactions_dict = {
        "user_id": {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0,
            "5": 1,
            "6": 1,
            "7": 1,
            "8": 1,
            "9": 1,
            "10": 2,
            "11": 2,
            "12": 2,
            "13": 2,
            "14": 2,
            "15": 3,
            "16": 3,
            "17": 3,
            "18": 3,
            "19": 3,
            "20": 4,
        },
        "item_id": {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 2,
            "4": 3,
            "5": 0,
            "6": 1,
            "7": 2,
            "8": 5,
            "9": 3,
            "10": 0,
            "11": 1,
            "12": 2,
            "13": 4,
            "14": 3,
            "15": 7,
            "16": 1,
            "17": 2,
            "18": 6,
            "19": 3,
            "20": 8,
        },
        "time": {
            "0": 1453939200,
            "1": 1453039200,
            "2": 1453032200,
            "3": 1453132200,
            "4": 1453132200,
            "5": 1453939201,
            "6": 1453039202,
            "7": 1453032203,
            "8": 1453132204,
            "9": 1453132205,
            "10": 1453939206,
            "11": 1453039207,
            "12": 1453032208,
            "13": 1453132209,
            "14": 1453132210,
            "15": 1453939211,
            "16": 1453039212,
            "17": 1453032213,
            "18": 1453132214,
            "19": 1453132215,
            "20": 1453939216,
        },
    }
    transactions_df = pd.DataFrame(transactions_dict)
    return transactions_df


def create_graph_custom():
    """Example function for creatin a custom graph for testing purposes. Only for fixtures reconstruction purposes."""
    graph_path = "src/tests/fixtures/graphs/graph_custom.pkl"
    transactions_custom = mapped_transactions_custom()
    graph_custom = generate_graph_dgsr(transactions_custom)
    with open(graph_path, "wb") as f:
        pickle.dump(graph_custom, f)


@pytest.fixture
def graph_custom():
    graph_path = "src/tests/fixtures/graphs/graph_custom.pkl"
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    return graph
