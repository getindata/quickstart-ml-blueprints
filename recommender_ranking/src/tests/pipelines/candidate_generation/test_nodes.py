import pandas as pd
import pytest

from src.gid_ml_framework.pipelines.candidate_generation.nodes import (
    _filter_transactions_by_bin,
    _most_sold_articles,
    _update_dict_of_sets,
    assign_global_articles,
    assign_segment_articles,
    collect_all_candidates,
    collect_previously_bought_articles,
    collect_previously_bought_prod_name_articles,
    segment_by_customer_age,
)


class TestGlobalArticles:
    # _most_sold_articles()
    def test_most_sold_articles(self, transactions_dummy_df):
        result_set = _most_sold_articles(transactions_dummy_df, top_k=4)
        expected_set = set(["0821057001", "0547780025", "0744934007", "0875899001"])
        assert result_set == expected_set

    def test_most_sold_articles_k_equals_zero(self, transactions_dummy_df):
        result_set = _most_sold_articles(transactions_dummy_df, top_k=0)
        expected_set = set()
        assert result_set == expected_set

    # assign_global_articles()
    def test_assign_global_articles(self, customers_dummy_df):
        articles_set = set(["0821057001", "0547780025", "0744934007", "0875899001"])
        result_df = assign_global_articles(customers_dummy_df, articles_set)
        assert len(result_df) == len(customers_dummy_df)
        assert "customer_id" in result_df.columns
        for _, value in result_df["global_articles"].iteritems():
            assert set(value) == articles_set


class TestSegmentArticles:
    # segment_by_customer_age()
    def test_segment_by_customer_bin_int(self, customers_dummy_df):
        n_bins = 2
        result_df = segment_by_customer_age(customers_dummy_df, n_bins=n_bins)
        assert len(customers_dummy_df) == len(result_df)
        assert "segment_bin" in result_df.columns
        assert n_bins == result_df["segment_bin"].nunique()

    def test_segment_by_customer_bin_more_bins_than_customers(self, customers_dummy_df):
        n_bins = 999
        with pytest.raises(ValueError):
            segment_by_customer_age(customers_dummy_df, n_bins=n_bins)

    def test_segment_by_customer_bin_zero_bins(self, customers_dummy_df):
        n_bins = 0
        result_df = segment_by_customer_age(customers_dummy_df, n_bins=n_bins)
        assert len(customers_dummy_df) == len(result_df)
        assert "segment_bin" in result_df.columns
        assert 1 == result_df["segment_bin"].nunique()

    def test_segment_by_customer_bin_negative_bins(self, customers_dummy_df):
        n_bins = -5
        with pytest.raises(ValueError):
            segment_by_customer_age(customers_dummy_df, n_bins=n_bins)

    def test_segment_by_customer_bin_float_list(self, customers_dummy_df):
        n_bins = [0.2, 0.5, 0.7, 0.9]
        result_df = segment_by_customer_age(customers_dummy_df, n_bins=n_bins)
        assert len(customers_dummy_df) == len(result_df)
        assert "segment_bin" in result_df.columns
        assert len(n_bins) == result_df["segment_bin"].nunique()

    def test_segment_by_customer_bin_negative_float_list(self, customers_dummy_df):
        n_bins = [-0.8, 0.2, 0.5, 0.7, 0.9]
        with pytest.raises(ValueError):
            segment_by_customer_age(customers_dummy_df, n_bins=n_bins)

    # _filter_transactions_by_bin()
    def test_filter_transactions_by_bin(
        self, transactions_dummy_df, customers_dummy_df
    ):
        segment_bin = "00"
        initial_customers_df = customers_dummy_df.copy()
        initial_customers_df["segment_bin"] = [
            "01",
            "02",
            "00",
            "00",
            "01",
            "02",
            "00",
            "02",
            "02",
            "00",
        ]
        result_df = _filter_transactions_by_bin(
            transactions_dummy_df, initial_customers_df, bin=segment_bin
        )
        assert len(result_df) < len(transactions_dummy_df)

        all_customers_from_bin = set(
            list(
                initial_customers_df[initial_customers_df["segment_bin"] == segment_bin]
                .loc[:, "customer_id"]
                .unique()
            )
        )
        result_customers = set(list(result_df.loc[:, "customer_id"].unique()))
        assert result_customers.intersection(all_customers_from_bin) == result_customers
        assert len(all_customers_from_bin) > 0

    def test_filter_transactions_by_bin_empty_bin(
        self, transactions_dummy_df, customers_dummy_df
    ):
        segment_bin = "99"
        initial_customers_df = customers_dummy_df.copy()
        initial_customers_df["segment_bin"] = [
            "01",
            "02",
            "00",
            "00",
            "01",
            "02",
            "00",
            "02",
            "02",
            "00",
        ]
        result_df = _filter_transactions_by_bin(
            transactions_dummy_df, initial_customers_df, bin=segment_bin
        )
        assert result_df.empty is True

    # _update_dict_of_sets()
    def test_update_dict_of_sets(self):
        cumulative_dict = {
            "00": set(["123", "12", "321"]),
            "01": set(["12", "1", "111"]),
        }
        new_dict = {"00": set(["1234", "43"]), "01": set(["5", "65"])}
        result_dict = _update_dict_of_sets(cumulative_dict, new_dict)
        expected_dict = {
            "00": set(["123", "12", "321", "1234", "43"]),
            "01": set(["12", "1", "111", "5", "65"]),
        }
        assert result_dict == expected_dict

    def test_update_dict_of_sets_empty_cumulative_dict(self):
        cumulative_dict = dict()
        new_dict = {"00": set(["1234", "43"]), "01": set(["5", "65"])}
        result_dict = _update_dict_of_sets(cumulative_dict, new_dict)
        expected_dict = new_dict
        assert result_dict == expected_dict

    def test_update_dict_of_sets_empty_new_dict(self):
        cumulative_dict = {"00": set(["1234", "43"]), "01": set(["5", "65"])}
        new_dict = dict()
        result_dict = _update_dict_of_sets(cumulative_dict, new_dict)
        expected_dict = cumulative_dict
        assert result_dict == expected_dict

    def test_update_dict_of_sets_intersection(self):
        cumulative_dict = {
            "00": set(["123", "12", "321"]),
            "01": set(["12", "1", "111"]),
        }
        new_dict = {
            "00": set(["123", "12", "321", "new"]),
            "01": set(["12", "1", "111", "new"]),
        }
        result_dict = _update_dict_of_sets(cumulative_dict, new_dict)
        expected_dict = new_dict
        assert result_dict == expected_dict

    def test_update_dict_of_sets_new_keys_in_new_dict(self):
        cumulative_dict = {
            "00": set(["123", "12", "321"]),
            "01": set(["12", "1", "111"]),
        }
        new_dict = {"98": set(["123", "12", "321"]), "99": set(["12", "1", "111"])}
        result_dict = _update_dict_of_sets(cumulative_dict, new_dict)
        expected_dict = {
            "00": set(["123", "12", "321"]),
            "01": set(["12", "1", "111"]),
            "98": set(["123", "12", "321"]),
            "99": set(["12", "1", "111"]),
        }
        assert result_dict == expected_dict

    # assign_segment_articles()
    def test_assign_segment_articles(self, customers_dummy_df):
        articles_dict = {
            "00": set(["111", "123"]),
            "01": set(["111", "101"]),
            "02": set(["321", "23"]),
        }
        initial_customers_df = customers_dummy_df.copy()
        initial_customers_df["segment_bin"] = [
            "01",
            "02",
            "00",
            "00",
            "01",
            "02",
            "00",
            "02",
            "02",
            "00",
        ]
        initial_customers_df = initial_customers_df[["customer_id", "segment_bin"]]
        result_df = assign_segment_articles(articles_dict, initial_customers_df)
        assert len(result_df) == len(initial_customers_df)
        assert set(["segment_articles", "customer_id"]) == set(result_df.columns)

    def test_assign_segment_articles_more_segments_in_dict(self, customers_dummy_df):
        articles_dict = {
            "00": set(["111", "123"]),
            "01": set(["111", "101"]),
            "02": set(["321", "23"]),
            "03": set(["111", "321", "22"]),
        }
        initial_customers_df = customers_dummy_df.copy()
        initial_customers_df["segment_bin"] = [
            "01",
            "02",
            "00",
            "00",
            "01",
            "02",
            "00",
            "02",
            "02",
            "00",
        ]
        initial_customers_df = initial_customers_df[["customer_id", "segment_bin"]]
        result_df = assign_segment_articles(articles_dict, initial_customers_df)
        assert len(result_df) == len(initial_customers_df)

    def test_assign_segment_articles_more_segments_customers(self, customers_dummy_df):
        articles_dict = {
            "00": set(["111", "123"]),
            "01": set(["111", "101"]),
        }
        initial_customers_df = customers_dummy_df.copy()
        initial_customers_df["segment_bin"] = [
            "01",
            "02",
            "00",
            "00",
            "01",
            "02",
            "00",
            "02",
            "02",
            "00",
        ]
        initial_customers_df = initial_customers_df[["customer_id", "segment_bin"]]
        result_df = assign_segment_articles(articles_dict, initial_customers_df)
        assert len(result_df) == 6


class TestPreviouslyBoughtArticles:
    # collect_previously_bought_articles()
    def test_collect_prev_bought(self, transactions_dummy_df):
        expected_no_customers = transactions_dummy_df["customer_id"].nunique()
        result_df = collect_previously_bought_articles(transactions_dummy_df)
        assert expected_no_customers == len(result_df)

    def test_collect_prev_bought_isin_col(self, transactions_dummy_df):
        result_df = collect_previously_bought_articles(transactions_dummy_df)
        assert "previously_bought" in result_df.columns
        assert "customer_id" in result_df.columns
        assert len(result_df.columns) == 2

    def test_collect_prev_bought_check_list_type(self, transactions_dummy_df):
        result_df = collect_previously_bought_articles(transactions_dummy_df)
        for _, value in result_df["previously_bought"].iteritems():
            assert isinstance(value, list)

    # collect_previously_bought_prod_name_articles()
    def test_collect_prev_bought_prod_name_isin_col(
        self, transactions_dummy_df, articles_dummy_df
    ):
        result_df = collect_previously_bought_prod_name_articles(
            transactions_dummy_df, articles_dummy_df
        )
        assert "previously_bought_prod_name" in result_df.columns
        assert "customer_id" in result_df.columns
        assert len(result_df.columns) == 2


class TestSimilarEmbeddingsArticles:
    pass


class TestCollectAllCandidates:
    def test_collect_all_candidates(self):
        global_articles = pd.DataFrame(
            data={
                "abc": ["123", "567", "469"],
                "bcd": ["123", "567", "469"],
                "cde": ["123", "567", "469"],
                "def": ["123", "567", "469"],
                "efg": ["123", "567", "469"],
            },
            columns=["customer_id", "global_articles"],
        )

        segment_articles = pd.DataFrame(
            data={
                "abc": ["123", "561", "560"],
                "bcd": ["459", "444", "421"],
                "cde": ["123", "561", "560"],
                "def": ["123", "561", "560"],
                "efg": ["023", "86", "555"],
            },
            columns=["customer_id", "segment_articles"],
        )

        prev_bought_articles = pd.DataFrame(
            data={"abc": ["53", "567", "469"], "def": ["23"]},
            columns=["customer_id", "previously_bought"],
        )

        prev_bough_prod_name = pd.DataFrame(
            data={
                "abc": ["470", "468", "467"],
                "def": ["24", "500", "100", "202", "256"],
            },
            columns=["customer_id", "previously_bought_prod_name"],
        )

        closest_image_embeddings = pd.DataFrame(
            data={"abc": ["25", "26", "0321"], "def": ["123", "400", "423", "432"]},
            columns=["customer_id", "clostest_image_embeddings"],
        )

        closest_text_embeddings = pd.DataFrame(
            data={"abc": ["15", "16"], "def": ["945", "1234", "3213"]},
            columns=["customer_id", "clostest_text_embeddings"],
        )

        result_df = collect_all_candidates(
            global_articles,
            segment_articles,
            prev_bought_articles,
            prev_bough_prod_name,
            closest_image_embeddings,
            closest_text_embeddings,
        )
        assert len(result_df) == len(global_articles)
