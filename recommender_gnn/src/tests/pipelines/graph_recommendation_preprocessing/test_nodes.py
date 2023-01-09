import numpy as np
import pandas as pd
import pytest

from recommender_gnn.pipelines.graph_recommendation_preprocessing.nodes import (
    _create_mapping,
    map_users_and_items,
    preprocess_transactions,
)


class TestConcatTrainVal:
    date_column = "date"
    column_names = {"item_id", "user_id", "time"}

    def test_given_empty_dataframes_should_raise_exception(self):
        empty_df = pd.DataFrame({})
        with pytest.raises(KeyError):
            preprocess_transactions(
                first_subset=empty_df,
                second_subset=empty_df,
                original_date_column=self.date_column,
            )

    def test_output_shape(self, bank_train_transactions, bank_val_transactions):
        df = preprocess_transactions(
            first_subset=bank_train_transactions,
            second_subset=bank_val_transactions,
            original_date_column=self.date_column,
        )
        expected_shape = (
            bank_train_transactions.shape[0] + bank_val_transactions.shape[0],
            bank_train_transactions.shape[1],
        )
        assert df.shape == expected_shape

    def test_output_columns(self, bank_train_transactions, bank_val_transactions):
        df = preprocess_transactions(
            first_subset=bank_train_transactions,
            second_subset=bank_val_transactions,
            original_date_column=self.date_column,
        )
        assert set(df.columns) == self.column_names


@pytest.mark.parametrize(
    "map_column",
    ["user_id", "item_id"],
)
class TestCreateMapping:
    def test_create_mapping_shapes(self, bank_concat_transactions, map_column):
        mapping = _create_mapping(bank_concat_transactions, map_column)
        assert len(bank_concat_transactions.loc[:, map_column].unique()) == len(mapping)

    def test_create_mapping_values(self, bank_concat_transactions, map_column):
        mapping = _create_mapping(bank_concat_transactions, map_column)
        assert set(mapping.values()) == set(np.arange(0, len(mapping)))

    def test_create_mapping_keys(self, bank_concat_transactions, map_column):
        mapping = _create_mapping(bank_concat_transactions, map_column)
        assert len(bank_concat_transactions.loc[:, map_column].unique()) == len(mapping)
        assert set(mapping.keys()) == set(bank_concat_transactions.loc[:, map_column])


class TestMapUsersAndItems:
    item_column = "item_id"
    user_column = "user_id"
    time_column = "time"

    def test_transactions_users(self, bank_concat_transactions):
        transactions_df, users_mapping, _ = map_users_and_items(
            bank_concat_transactions
        )
        assert set(transactions_df.loc[:, self.user_column]) == set(
            users_mapping.values()
        )

    def test_transactions_items(self, bank_concat_transactions):
        transactions_df, _, items_mapping = map_users_and_items(
            bank_concat_transactions
        )
        assert set(transactions_df.loc[:, self.item_column]) == set(
            items_mapping.values()
        )

    def test_transactions_structure(self, bank_concat_transactions):
        columns = {self.user_column, self.item_column, self.time_column}
        transactions_df, _, _ = map_users_and_items(bank_concat_transactions)
        assert transactions_df.shape == bank_concat_transactions.shape
        assert set(transactions_df.columns) == columns
