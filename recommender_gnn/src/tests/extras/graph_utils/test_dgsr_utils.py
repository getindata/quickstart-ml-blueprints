import os
from pathlib import Path

import pytest
from dgl import DGLHeteroGraph
from pytest_lazyfixture import lazy_fixture

from recommender_gnn.extras.graph_utils.dgsr_utils import (
    SubGraphsDataset,
    eval_metric,
    load_graphs_python,
    save_graphs_python,
)


class TestSubGraphsDataet:
    extension = "pkl"

    @pytest.mark.parametrize(
        "dir, expected_size",
        [
            (lazy_fixture("train_subgraphs_path"), 12),
            (lazy_fixture("val_subgraphs_path"), 4),
            (lazy_fixture("test_subgraphs_path"), 4),
            (lazy_fixture("predict_subgraphs_path"), 4),
        ],
    )
    def test_initialize_correct_dataset_size(self, dir, expected_size):
        dataset = SubGraphsDataset(dir, load_graphs_python, self.extension)
        assert len(dataset) == expected_size

    @pytest.mark.parametrize(
        "dir, expected_value",
        [
            (lazy_fixture("train_subgraphs_path"), "0_1"),
            (lazy_fixture("val_subgraphs_path"), "0_2"),
            (lazy_fixture("test_subgraphs_path"), "0_3"),
            (lazy_fixture("predict_subgraphs_path"), "0_4"),
        ],
    )
    def test_get_item_should_return_data_and_key(self, dir, expected_value):
        dataset = SubGraphsDataset(dir, load_graphs_python, self.extension)
        data, key = dataset.__getitem__(0)
        assert isinstance(data[0][0], DGLHeteroGraph)
        assert key == expected_value

    @pytest.mark.parametrize(
        "dir",
        [
            (lazy_fixture("train_subgraphs_path")),
            (lazy_fixture("val_subgraphs_path")),
            (lazy_fixture("test_subgraphs_path")),
            (lazy_fixture("predict_subgraphs_path")),
        ],
    )
    def test_get_graphs_collection_should_return_dict(self, dir):
        dataset = SubGraphsDataset(dir, load_graphs_python, self.extension)
        dataset_dict = dataset.get_graphs_collection()
        assert isinstance(dataset_dict, dict)
        assert len(dataset_dict) == len(dataset)

    def test_initialization_given_not_existing_dir(self):
        with pytest.raises(FileNotFoundError):
            SubGraphsDataset("not_existing_dir", load_graphs_python, self.extension)


def test_eval_metric_recall(scores_custom):
    results = [eval_metric([score]) for score in scores_custom]
    recalls = [result[2] for result in results]
    assert recalls == [0.1, 0.0, 0.1, 0.1, 0.0]


def test_eval_metric_ndgg(scores_custom):
    results = [eval_metric([score]) for score in scores_custom]
    ndggs = [result[5] for result in results]
    assert ndggs == [0.1, 0.0, 0.03562071871080222, 0.023981246656813147, 0.0]


def test_save_graphs_python_should_write_to_directory(tmp_path):
    graphs_collection = {}
    tmp_file_path = os.path.join(str(tmp_path), "graphs.pkl")
    save_graphs_python(tmp_file_path, graphs_collection)
    assert len(list(Path(tmp_path).iterdir())) == 1
