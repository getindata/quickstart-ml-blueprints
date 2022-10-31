import numpy as np
import pandas as pd
import pytest

from src.gid_ml_framework.pipelines.candidate_generation_validation.nodes import (
    _concatenate_all_lists,
    _get_candidate_column_names,
    _get_recall,
    _get_unique_transactions,
)


# _get_candidate_column_names()
def test_get_candidate_column_names(candidates_dummy_df):
    results_list = _get_candidate_column_names(candidates_dummy_df)
    expected_list = [
        "global_articles",
        "segment_articles",
        "previously_bought",
        "previously_bought_prod_name",
        "closest_image_embeddings",
        "closest_text_embeddings",
    ]
    assert set(results_list) == set(expected_list)


# _get_unique_transactions()
class TestGetUniqueTransactions:
    def test_get_unique_transactions_no_duplicates(self, transactions_dummy_df):
        initial_df = transactions_dummy_df
        result_df = _get_unique_transactions(transactions_dummy_df)
        assert len(initial_df) == len(result_df)

    @pytest.mark.parametrize("k", [2, 3])
    def test_get_unique_transactions_duplicates(self, transactions_dummy_df, k):
        initial_df = transactions_dummy_df
        list_of_dfs = [transactions_dummy_df for _ in range(k)]
        repeated_transactions = pd.concat(list_of_dfs, axis=1).reset_index(drop=True)
        result_df = _get_unique_transactions(repeated_transactions)
        assert len(initial_df) == len(result_df)


# _get_recall()
class TestGetRecall:
    def test_correct_float(self, candidates_dummy_df, transactions_dummy_df):
        candidate_col = "global_articles"
        result_recall = _get_recall(
            candidates_dummy_df, transactions_dummy_df, candidate_col
        )
        assert pytest.approx(result_recall) == 0.1

        candidate_col = "closest_image_embeddings"
        result_recall = _get_recall(
            candidates_dummy_df, transactions_dummy_df, candidate_col
        )
        assert pytest.approx(result_recall) == 0.0

    def test_returns_float(self, candidates_dummy_df, transactions_dummy_df):
        candidate_cols = [
            "global_articles",
            "segment_articles",
            "previously_bought",
            "previously_bought_prod_name",
            "closest_image_embeddings",
            "closest_text_embeddings",
        ]
        for candidate_col in candidate_cols:
            result_recall = _get_recall(
                candidates_dummy_df, transactions_dummy_df, candidate_col
            )
            assert result_recall >= 0 and result_recall <= 1

    def test_not_existing_column(self, candidates_dummy_df, transactions_dummy_df):
        candidate_col = "does_not_exist"
        with pytest.raises(KeyError):
            _get_recall(candidates_dummy_df, transactions_dummy_df, candidate_col)

    def test_non_overlapping_records(self, candidates_dummy_df, transactions_dummy_df):
        candidate_col = "global_articles"
        # overwriting transactions, so candidates won't have any common records with transactions
        initial_transactions_df = transactions_dummy_df
        initial_transactions_df["customer_id"] = range(20)
        initial_transactions_df["customer_id"] = initial_transactions_df[
            "customer_id"
        ].astype(str)
        initial_transactions_df["article_id"] = range(20, 40)
        initial_transactions_df["article_id"] = initial_transactions_df[
            "article_id"
        ].astype(str)
        result_recall = _get_recall(
            candidates_dummy_df, initial_transactions_df, candidate_col
        )
        assert pytest.approx(result_recall) == 0.0


class TestConcatenateAllLists:
    def test_two_arrays(self):
        no_cols = 2
        row = [
            np.array(["123", "12", "1"], dtype=object),
            np.array(["321", "32", "3"], dtype=object),
        ]
        result_set = _concatenate_all_lists(row, no_cols)
        expected_set = {"123", "12", "1", "321", "32", "3"}
        assert len(result_set) == 6
        assert result_set == expected_set

    def test_single_array(self):
        pass

    def test_null_candidates_items(self):
        no_cols = 3
        row = [
            np.array(["123", "12", "1"], dtype=object),
            None,
            np.array(["321", "32", "3"], dtype=object),
        ]
        result_set = _concatenate_all_lists(row, no_cols)
        expected_set = {"123", "12", "1", "321", "32", "3"}
        assert len(result_set) == 6
        assert result_set == expected_set

    def test_only_null_candidates_items(self):
        no_cols = 3
        row = [None, None, None]
        result_set = _concatenate_all_lists(row, no_cols)
        assert bool(result_set) is False

    def test_repeating_candidates_items(self):
        no_cols = 3
        row = [
            np.array(["123", "12", "1"], dtype=object),
            np.array(["123", "12", "1"], dtype=object),
            np.array(["123", "12", "1"], dtype=object),
        ]
        result_set = _concatenate_all_lists(row, no_cols)
        expected_set = {"123", "12", "1"}
        assert len(result_set) == 3
        assert result_set == expected_set
