import numpy as np
import pytest

from src.gid_ml_framework.pipelines.candidates_feature_engineering.nodes import (
    _assing_multiple_strategies,
    _cosine_similarity,
    _jaccard_similarity,
    create_list_of_previously_bought_articles,
    create_set_of_attributes,
    filter_last_n_rows_per_customer,
    unpack_candidates,
)


class TestFilterLastNRowsPerCustomer:
    def test_standard_n(self, transactions_dummy_df):
        n = 2
        initial_df = transactions_dummy_df.copy()
        result_df = filter_last_n_rows_per_customer(transactions_dummy_df, n, "t_dat")
        assert len(initial_df) > len(result_df)
        assert initial_df["customer_id"].nunique() == result_df["customer_id"].nunique()

    def test_negative_n(self, transactions_dummy_df):
        n = -5
        result_df = filter_last_n_rows_per_customer(transactions_dummy_df, n, "t_dat")
        assert result_df.empty is True

    def test_n_equals_zero(self, transactions_dummy_df):
        n = 0
        result_df = filter_last_n_rows_per_customer(transactions_dummy_df, n, "t_dat")
        assert result_df.empty is True

    def test_big_n(self, transactions_dummy_df):
        n = 999_999
        initial_df = transactions_dummy_df.copy()
        result_df = filter_last_n_rows_per_customer(transactions_dummy_df, n, "t_dat")
        assert initial_df.shape == result_df.shape


class TestUnpackingCandidates:
    def test_assign_multiple_strategies(self, long_candidates_dummy_df):
        initial_df = long_candidates_dummy_df.copy()
        result_df = _assing_multiple_strategies(long_candidates_dummy_df)
        assert len(result_df) < len(initial_df)
        assert "multiple_strategies" not in initial_df["strategy_name"].unique()
        assert "multiple_strategies" in result_df["strategy_name"].unique()

    def test_unpack_candidates_columns(self, candidates_dummy_df):
        result_df = unpack_candidates(candidates_dummy_df)
        expected_cols = set(["customer_id", "article_id", "strategy_name"])
        assert expected_cols == set(list(result_df.columns))

    def test_unpack_candidates(self, candidates_dummy_df):
        initial_df = candidates_dummy_df.copy()
        result_df = unpack_candidates(candidates_dummy_df)
        assert len(result_df) > len(initial_df)


class TestJaccardSimilarity:
    # _jaccard_similarity()
    def test_empty_sets(self):
        x = set()
        y = set()
        with pytest.raises(ZeroDivisionError):
            _jaccard_similarity(x, y)

    def test_one_set_empty(self):
        x = set()
        y = set(["A", "B"])
        result = _jaccard_similarity(x, y)
        assert np.isclose(result, 0)

    def test_one_common_item(self):
        x = set(["A", "B"])
        y = set(["B", "C"])
        result = _jaccard_similarity(x, y)
        assert np.isclose(result, 1 / 3)

    def test_no_common_items(self):
        x = set(["A", "B"])
        y = set(["C", "D"])
        result = _jaccard_similarity(x, y)
        assert np.isclose(result, 0)

    def test_all_common_items(self):
        x = set(["A", "B"])
        y = x
        result = _jaccard_similarity(x, y)
        assert np.isclose(result, 1)

    # create_set_of_attributes()
    def test_create_set_of_attributes_cols(self, articles_dummy_df):
        article_cols = [
            "colour_group_name",
            "department_name",
            "garment_group_name",
            "product_type_name",
        ]
        result_df = create_set_of_attributes(articles_dummy_df, article_cols)
        assert "set_of_attributes" in result_df.columns
        assert len(result_df.columns) == 1

    def test_each_row_is_a_set(self, articles_dummy_df):
        article_cols = [
            "colour_group_name",
            "department_name",
            "garment_group_name",
            "product_type_name",
        ]
        result_df = create_set_of_attributes(articles_dummy_df, article_cols)
        for _, row in result_df.iterrows():
            assert isinstance(row[0], set)

    def test_not_existing_cols(self, articles_dummy_df):
        article_cols = [
            "does_not_exist",
            "department_name",
            "garment_group_name",
            "product_type_name",
        ]
        with pytest.raises(KeyError):
            create_set_of_attributes(articles_dummy_df, article_cols)

    # create_list_of_previously_bought_articles()
    def test_previously_bought_articles_cols(self, transactions_dummy_df):
        result_df = create_list_of_previously_bought_articles(transactions_dummy_df)
        assert "list_of_articles" in result_df.columns
        assert len(result_df.columns) == 1

    def test_previously_bought_articles_aggregation(self, transactions_dummy_df):
        initial_df = transactions_dummy_df.copy()
        result_df = create_list_of_previously_bought_articles(transactions_dummy_df)
        assert len(initial_df) > len(result_df)

    def test_each_row_is_a_list(self, transactions_dummy_df):
        result_df = create_list_of_previously_bought_articles(transactions_dummy_df)
        for _, row in result_df.iterrows():
            assert isinstance(row[0], list)


class TestCosineSimilarity:
    def test_same_vectors(self):
        A = np.array([1, 5, 10, 20])
        B = A
        result = _cosine_similarity(A, B)
        assert np.isclose(result, 1.0)

    def test_opposite_vectors(self):
        A = np.array([1, 5, 10, 20])
        B = A * (-1)
        result = _cosine_similarity(A, B)
        assert np.isclose(result, -1.0)

    def test_orthogonal_vectors(self):
        A = np.array([1, 5, 10, 20])
        B = np.array([-5, 1, -20, 10])
        result = _cosine_similarity(A, B)
        assert np.isclose(result, 0.0)

        A = np.array([1, 1, 5, 2])
        B = np.array([-1, 1, 0, 0])
        result = _cosine_similarity(A, B)
        assert np.isclose(result, 0.0)
