import pytest
from pytest_lazyfixture import lazy_fixture

from recommender_gnn.extras.datasets.graph_dataset import DGSRSubGraphsDataSet
from recommender_gnn.extras.graph_utils.dgsr_utils import SubGraphsDataset


class TestDGSRSubGraphsDataSet:
    save_args = {"file_extension": "pkl"}

    @pytest.mark.parametrize(
        "dir",
        [
            (lazy_fixture("train_subgraphs_path")),
            (lazy_fixture("val_subgraphs_path")),
            (lazy_fixture("test_subgraphs_path")),
            (lazy_fixture("predict_subgraphs_path")),
        ],
    )
    def test_load_should_return_sub_graphs_dataset(self, dir):
        dataset = DGSRSubGraphsDataSet(dir, self.save_args)
        dataset_loaded = dataset._load()
        assert isinstance(dataset_loaded, SubGraphsDataset)

    @pytest.mark.parametrize(
        "dir, expected_size",
        [
            (lazy_fixture("train_subgraphs_path"), 12),
            (lazy_fixture("val_subgraphs_path"), 4),
            (lazy_fixture("test_subgraphs_path"), 4),
            (lazy_fixture("predict_subgraphs_path"), 4),
        ],
    )
    def test_load_correct_dataset_size(self, dir, expected_size):
        dataset = DGSRSubGraphsDataSet(dir, self.save_args)
        dataset_loaded = dataset._load()
        assert len(dataset_loaded) == expected_size

    def test_load_given_wrong_dir_should_raise_exception(self, tmp_path):
        wrong_dir = str(tmp_path)
        with pytest.raises(FileNotFoundError):
            dataset = DGSRSubGraphsDataSet(wrong_dir, self.save_args)
            dataset._load()

    @pytest.mark.parametrize(
        "test_list",
        [
            (lazy_fixture("train_subgraphs_list")),
            (lazy_fixture("val_subgraphs_list")),
            (lazy_fixture("test_subgraphs_list")),
            (lazy_fixture("predict_subgraphs_list")),
        ],
    )
    def test_unpack_data_should_return_graphs_collection_dictionary(self, test_list):
        graphs_collection = DGSRSubGraphsDataSet._unpack_data(test_list)
        assert isinstance(graphs_collection, dict)
        assert len(graphs_collection) == len(test_list)
