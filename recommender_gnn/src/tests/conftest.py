import pickle

import pandas as pd
import pytest

from recommender_gnn.pipelines.graph_recommendation_modeling.nodes import (
    generate_graph_dgsr,
    preprocess_dgsr,
)


def create_mapped_transactions_custom():
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
            "3": 8,
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


@pytest.fixture
def mapped_transactions_custom():
    return create_mapped_transactions_custom()


def create_graph_custom():
    """Example function for creating a custom graph for testing purposes. Only for fixtures reconstruction purposes."""
    graph_path = "src/tests/fixtures/graphs/graph_custom.pkl"
    transactions_custom = mapped_transactions_custom()
    graph_custom = generate_graph_dgsr(transactions_custom)
    with open(graph_path, "wb") as f:
        pickle.dump(graph_custom, f, protocol=-1)


def load_graph_custom():
    graph_path = "src/tests/fixtures/graphs/graph_custom.pkl"
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    return graph


@pytest.fixture
def graph_custom():
    graph = load_graph_custom()
    return graph


def create_subgraphs_lists_custom():
    """Example function for creating a custom train/val/test/predict subgraphs lists for testing purposes. Only
    for fixtures reconstruction purposes."""
    transactions_custom = create_mapped_transactions_custom()
    full_graph_custom = load_graph_custom()
    _, val_list, test_list, _ = preprocess_dgsr(
        transactions_custom,
        full_graph_custom,
        50,
        50,
        3,
        True,
        True,
        False,
    )
    train_list, _, _, predict_list = preprocess_dgsr(
        transactions_custom,
        full_graph_custom,
        50,
        50,
        3,
        False,
        False,
        True,
    )
    subsets = [train_list, val_list, test_list, predict_list]
    subnames = ["train", "val", "test", "predict"]
    return subsets, subnames


def get_subset_subgraphs_path(subset: str) -> str:
    subgraph_path = f"src/tests/fixtures/graphs/{subset}_subgraphs"
    return subgraph_path


@pytest.fixture
def train_subgraphs_path():
    subgraphs = get_subset_subgraphs_path("train")
    return subgraphs


@pytest.fixture
def val_subgraphs_path():
    subgraphs = get_subset_subgraphs_path("val")
    return subgraphs


@pytest.fixture
def test_subgraphs_path():
    subgraphs = get_subset_subgraphs_path("test")
    return subgraphs


@pytest.fixture
def predict_subgraphs_path():
    subgraphs = get_subset_subgraphs_path("predict")
    return subgraphs


@pytest.fixture
def otto_dummy_df():
    path = "src/tests/fixtures/dataframes/otto_raw_sample.csv"
    otto_df = pd.read_csv(path)
    return otto_df
