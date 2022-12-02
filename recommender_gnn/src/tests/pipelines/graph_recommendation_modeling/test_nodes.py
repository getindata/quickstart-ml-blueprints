import pandas as pd
import pytest

from recommender_gnn.pipelines.graph_recommendation_modeling.nodes import (
    _get_sub_edges,
    _prepare_user_data,
    _preprocess_transactions,
    generate_graph_dgsr,
    preprocess_dgsr,
    sample_negatives_dgsr,
)


class TestSampleNegativesDgsr:
    user_column = "user_id"
    item_column = "item_id"

    def test_given_empty_df_should_raise_exception(self):
        empty_df = pd.DataFrame({})
        with pytest.raises(KeyError):
            sample_negatives_dgsr(empty_df)

    def test_shape_and_values(self, mapped_transactions):
        negatives = sample_negatives_dgsr(mapped_transactions)
        unique_users = mapped_transactions.loc[:, self.user_column].unique()
        assert negatives.shape[0] == len(unique_users)
        assert set(negatives.index) == set(unique_users)

    def test_if_negative_samples_not_present_in_data(self, mapped_transactions_small):
        negatives = sample_negatives_dgsr(mapped_transactions_small)
        lsuffix, rsuffix = "_left", "_right"
        joined_df = mapped_transactions_small.join(
            negatives, on=self.user_column, lsuffix=lsuffix, rsuffix=rsuffix
        )
        left_item_column = "".join([self.item_column, lsuffix])
        right_item_column = "".join([self.item_column, rsuffix])
        check_inclusion = joined_df.loc[:, left_item_column].isin(
            joined_df.loc[:, right_item_column]
        )
        assert not check_inclusion.sum()


class TestGenerateGraphDgsr:
    user_column = "user_id"
    user_node_name = "user"
    item_column = "item_id"
    item_node_name = "item"
    time_column = "time"

    @staticmethod
    def _test_inclusion_of_transaction_entities_in_graph(
        node_of_entity, entity_column, mapped_transactions_custom
    ):
        graph = generate_graph_dgsr(mapped_transactions_custom)
        graph_entities = set(graph.nodes[node_of_entity].data[entity_column])
        df_entities = set(mapped_transactions_custom.loc[:, entity_column].unique())
        assert not graph_entities.intersection(df_entities)

    def test_preprocess_transactions_should_return_expected_timestamps(
        self, mapped_transactions_custom
    ):
        preprocessed_trans = _preprocess_transactions(
            mapped_transactions_custom.iloc[:5, :]
        )
        timestamps = set(preprocessed_trans.loc[:, self.time_column])
        assert timestamps == {
            1453939200,
            1453039200,
            1453032200,
            1453132201,
            1453132200,
        }

    def test_users_inclusion_of_transaction_users_in_graph(
        self, mapped_transactions_custom
    ):
        self._test_inclusion_of_transaction_entities_in_graph(
            self.user_node_name, self.user_column, mapped_transactions_custom
        )

    def test_items_inclusion_of_transaction_items_in_graph(
        self, mapped_transactions_custom
    ):
        self._test_inclusion_of_transaction_entities_in_graph(
            self.item_node_name, self.item_column, mapped_transactions_custom
        )

    def test_number_edges_in_graphs_equals_number_transactions(
        self, mapped_transactions_custom
    ):
        graph = generate_graph_dgsr(mapped_transactions_custom)
        n_edges = graph.number_of_edges() / 2
        n_transactions = mapped_transactions_custom.shape[0]
        assert n_edges == n_transactions


class TestPreprocessDgsr:
    item_max_length = 50
    user_max_length = 50
    k_hop = 3
    user = 0
    user_time = [1453032200, 1453039200, 1453132200, 1453132200, 1453939200]
    user_seq = [2, 1, 8, 3, 0]

    def test_prepare_user_data(self, mapped_transactions_custom):
        user_seq, user_time = _prepare_user_data(mapped_transactions_custom, self.user)
        assert list(user_seq) == self.user_seq
        assert list(user_time) == self.user_time

    @pytest.mark.parametrize(
        "predict_condition, start_t, next_index, expected_result",
        [
            (True, 1, 3, 9),
            (False, 1, 3, 8),
            (True, 0, 4, 17),
            (False, 0, 4, 16),
        ],
    )
    def test_get_sub_edges(
        self, graph_custom, predict_condition, start_t, next_index, expected_result
    ):
        sub_u_eid, sub_i_eid = _get_sub_edges(
            graph_custom, self.user_time, predict_condition, start_t, next_index
        )
        result = sum(sub_u_eid + sub_i_eid)
        assert result == expected_result

    @pytest.mark.parametrize(
        "val_flag, test_flag, predict_flag, expected_result",
        [
            (True, True, False, (4, 4, 4, 0)),
            (False, False, True, (12, 0, 0, 4)),
            (True, False, False, (8, 4, 0, 0)),
            (False, False, False, (12, 0, 0, 0)),
        ],
    )
    def test_preprocess_dgsr(
        self,
        mapped_transactions_custom,
        graph_custom,
        val_flag,
        test_flag,
        predict_flag,
        expected_result,
    ):
        train_list, val_list, test_list, predict_list = preprocess_dgsr(
            mapped_transactions_custom,
            graph_custom,
            self.item_max_length,
            self.user_max_length,
            self.k_hop,
            val_flag,
            test_flag,
            predict_flag,
        )
        assert (
            len(train_list),
            len(val_list),
            len(test_list),
            len(predict_list),
        ) == expected_result
