import pandas as pd
import pytest

from recommender_gnn.pipelines.graph_recommendation_modeling.nodes import (
    generate_graph_dgsr,
    sample_negatives_dgsr,
)


class TestSampleNegativesDgsr:
    user_column = "user_id"
    item_column = "item_id"

    def test_given_empty_df(self):
        empty_df = pd.DataFrame({})
        with pytest.raises(KeyError):
            sample_negatives_dgsr(empty_df)

    def test_structure(self, mapped_transactions):
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

    def test_users_inclusion(self, mapped_transactions_custom):
        graph = generate_graph_dgsr(mapped_transactions_custom)
        graph_users = set(graph.nodes["user"].data[self.user_column])
        df_users = set(mapped_transactions_custom.loc[:, self.user_column].unique())
        assert not graph_users.intersection(df_users)
