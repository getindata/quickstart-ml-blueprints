import pytest
from dgl.heterograph import DGLHeteroGraph
from pytest_lazyfixture import lazy_fixture

from recommender_gnn.gnn_models.recommendation.dgsr import DGSR
from recommender_gnn.pipelines.graph_recommendation.nodes import (
    _get_data_stats,
    _get_loaders,
    _get_model,
)


class TestGetDataStats:
    def test_get_data_stats_should_return_expected_values(
        self, mapped_transactions_custom, train_subgraphs
    ):
        data_stats = _get_data_stats(mapped_transactions_custom, train_subgraphs)
        assert data_stats == (5, 9)

    @staticmethod
    def _check_entity_type_sample_count_should_not_be_greater_than_overall_count(
        mapped_transactions_custom, subset, entity_type
    ):
        data_stats = _get_data_stats(mapped_transactions_custom, subset)
        enitity_overall_size = data_stats[0] if entity_type == "user" else data_stats[1]
        subset_dict = subset.get_graphs_collection()
        entity_nodes = [
            subgprah[0][0].num_nodes(entity_type)
            for subgprah in list(subset_dict.values())
        ]
        entity_size_check = [
            user_node > enitity_overall_size for user_node in entity_nodes
        ]
        assert not sum(entity_size_check)

    @pytest.mark.parametrize(
        "subset",
        [
            (lazy_fixture("train_subgraphs")),
            (lazy_fixture("val_subgraphs")),
            (lazy_fixture("test_subgraphs")),
            (lazy_fixture("predict_subgraphs")),
        ],
    )
    def _check_user_sample_count_should_not_be_greater_than_overall_count(
        self, mapped_transactions_custom, subset
    ):
        self._check_entity_type_sample_count_should_not_be_greater_than_overall_count(
            mapped_transactions_custom, subset, "user"
        )

    @pytest.mark.parametrize(
        "subset",
        [
            (lazy_fixture("train_subgraphs")),
            (lazy_fixture("val_subgraphs")),
            (lazy_fixture("test_subgraphs")),
            (lazy_fixture("predict_subgraphs")),
        ],
    )
    def _check_item_sample_count_should_not_be_greater_than_overall_count(
        self, mapped_transactions_custom, subset
    ):
        self._check_entity_type_sample_count_should_not_be_greater_than_overall_count(
            mapped_transactions_custom, subset, "item"
        )


class TestGetLoaders:
    data_stats = (5, 9)

    @pytest.mark.parametrize(
        "subset, expected_size",
        [
            (lazy_fixture("train_subgraphs"), 12),
            (lazy_fixture("val_subgraphs"), 4),
            (lazy_fixture("test_subgraphs"), 4),
            (lazy_fixture("predict_subgraphs"), 4),
        ],
    )
    def test_get_train_loader_size(self, subset, train_params_custom, expected_size):
        train_loader, _, _ = _get_loaders(
            train_set=subset,
            train_params=train_params_custom,
            data_stats=self.data_stats,
        )
        assert len(train_loader) == expected_size

    def test_get_train_loader_next_item_type(
        self, train_subgraphs, train_params_custom
    ):
        train_loader, _, _ = _get_loaders(
            train_set=train_subgraphs,
            train_params=train_params_custom,
            data_stats=self.data_stats,
        )
        item = train_loader.dataset[0][0][0][0]
        assert isinstance(item, DGLHeteroGraph)

    @pytest.mark.parametrize(
        "subset, expected_size",
        [
            (lazy_fixture("train_subgraphs"), 12),
            (lazy_fixture("val_subgraphs"), 4),
            (lazy_fixture("test_subgraphs"), 4),
            (lazy_fixture("predict_subgraphs"), 4),
        ],
    )
    def test_get_validation_loader_size(
        self, subset, train_params_custom, expected_size, negatives_custom
    ):
        _, validation_loader, _ = _get_loaders(
            val_set=subset,
            train_params=train_params_custom,
            data_stats=self.data_stats,
            negative_samples=negatives_custom,
        )
        assert len(validation_loader) == expected_size


def test_get_model_should_return_dgsr_model(model_params_custom, train_params_custom):
    device = "cpu"
    data_stats = (5, 9)
    model = _get_model(device, model_params_custom, train_params_custom, data_stats)
    assert isinstance(model, DGSR)
