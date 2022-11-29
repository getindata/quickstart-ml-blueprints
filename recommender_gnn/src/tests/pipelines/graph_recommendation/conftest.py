import pandas as pd
import pytest

from recommender_gnn.extras.datasets.graph_dataset import DGSRSubGraphsDataSet
from recommender_gnn.extras.graph_utils.dgsr_utils import (
    SubGraphsDataset,
    load_graphs_python,
)
from recommender_gnn.pipelines.graph_recommendation_modeling.nodes import (
    preprocess_dgsr,
    sample_negatives_dgsr,
)
from tests.conftest import graph_custom, mapped_transactions_custom


def create_subgraphs_custom():
    """Example function for creating a custom train/val/test/predict subgraphs for testing purposes. Only for fixtures
    reconstruction purposes."""
    transactions_custom = mapped_transactions_custom()
    full_graph_custom = graph_custom()
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
    for subname, subset in zip(subnames, subsets):
        save_args = {"file_extension": "pkl"}
        subset_dataset = DGSRSubGraphsDataSet(
            f"src/tests/fixtures/graphs/{subname}_subgraphs", save_args
        )
        subset_dataset._save(subset)


def create_negatives_custom():
    """Example function for creating dataframe with negative samples for testing purposes. Only for fixtures
    reconstruction purposes."""
    transactions_custom = mapped_transactions_custom()
    negatives = sample_negatives_dgsr(transactions_custom)
    negatives.to_csv("src/tests/fixtures/dataframes/negatives.csv", index=False)


@pytest.fixture
def negatives_custom():
    return pd.read_csv("src/tests/fixtures/dataframes/negatives.csv")


@pytest.fixture
def train_params_custom():
    return {
        "batch_size": 1,
        "epochs": 2,
        "k": 20,
        "l2": 0.0001,
        "lr": 0.001,
        "validate": True,
    }


@pytest.fixture
def model_params_custom():
    return {
        "attn_drop_out": 0.3,
        "feat_drop_out": 0.3,
        "hidden_size": 50,
        "item_long": "orgat",
        "item_max_length": 50,
        "item_short": "att",
        "item_update": "rnn",
        "k_hop": 2,
        "layer_num": 3,
        "user_long": "orgat",
        "user_max_length": 50,
        "user_short": "att",
        "user_update": "rnn",
    }


def subgraphs_subset(subset: str) -> SubGraphsDataset:
    subgraph_path = f"src/tests/fixtures/graphs/{subset}_subgraphs"
    subgraphs = SubGraphsDataset(subgraph_path, load_graphs_python, "pkl")
    return subgraphs


@pytest.fixture
def train_subgraphs():
    subgraphs = subgraphs_subset("train")
    return subgraphs


@pytest.fixture
def val_subgraphs():
    subgraphs = subgraphs_subset("val")
    return subgraphs


@pytest.fixture
def test_subgraphs():
    subgraphs = subgraphs_subset("test")
    return subgraphs


@pytest.fixture
def predict_subgraphs():
    subgraphs = subgraphs_subset("predict")
    return subgraphs
