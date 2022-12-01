import pickle

import pytest

from tests.conftest import create_subgraphs_lists_custom


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


def create_subgraphs_lists_pickles_custom():
    """Example function for creating a custom train/val/test/predict subgraphs lists pickles for testing purposes. Only
    for fixtures reconstruction purposes."""
    subsets, subnames = create_subgraphs_lists_custom()
    sub_dict = dict(zip(subnames, subsets))
    for subname in subnames:
        save_path = f"src/tests/fixtures/graphs/{subname}_subgraphs_lists.pkl"
        with open(save_path, "wb") as file:
            pickle.dump(sub_dict[subnames], file, protocol=-1)


def read_subgraphs_list(subset: str) -> str:
    subgraph_list_path = f"src/tests/fixtures/graphs/{subset}_subgraphs_lists.pkl"
    with open(subgraph_list_path, "rb") as file:
        subgraphs_list = pickle.load(file)
    return subgraphs_list


@pytest.fixture
def train_subgraphs_list():
    subgraphs_list = read_subgraphs_list("train")
    return subgraphs_list


@pytest.fixture
def val_subgraphs_list():
    subgraphs_list = read_subgraphs_list("val")
    return subgraphs_list


@pytest.fixture
def test_subgraphs_list():
    subgraphs_list = read_subgraphs_list("test")
    return subgraphs_list


@pytest.fixture
def predict_subgraphs_list():
    subgraphs_list = read_subgraphs_list("predict")
    return subgraphs_list
