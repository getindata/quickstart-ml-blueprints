import pytest

from src.gid_ml_framework.pipelines.feature_engineering_manual.nodes import (
    _add_suffix_except_col,
    _average_purchase_span,
    _concat_dataframes_on_index,
    _count_of_article_id_per_customer_id,
    _count_of_article_id_per_customer_product_group,
    _count_of_product_group_name_per_customer_id,
    _days_since_first_transactions,
    _days_since_last_transactions,
    _mean_perc_sales_channel_id,
    _perc_sales_channel_id,
    _rebuying_articles,
    create_article_features,
    create_customer_features,
    create_customer_product_group_features,
)


# _concat_dataframes_on_index()
def test_concat_dataframes_on_single_index(feature_one_dummy_df, feature_two_dummy_df):
    initial_one_ids = feature_one_dummy_df.id.unique()
    initial_two_ids = feature_two_dummy_df.id.unique()
    assert set(initial_one_ids) == set(initial_two_ids)

    result_df = _concat_dataframes_on_index(
        [feature_one_dummy_df, feature_two_dummy_df], index_name="id"
    )
    assert set(result_df.index.to_list()) == set(initial_one_ids)
    assert len(feature_one_dummy_df) == len(feature_two_dummy_df) == len(result_df)
    assert result_df.isna().values.sum() == 0


def test_concat_dataframes_on_double_index(feature_one_dummy_df, feature_two_dummy_df):
    initial_one_df = feature_one_dummy_df
    initial_two_df = feature_two_dummy_df
    second_index = range(len(feature_one_dummy_df))
    initial_one_df["second_id"] = second_index
    initial_two_df["second_id"] = second_index
    result_df = _concat_dataframes_on_index(
        [initial_one_df, initial_two_df], index_name=["id", "second_id"]
    )
    assert len(initial_one_df) == len(initial_two_df) == len(result_df)
    assert result_df.isna().values.sum() == 0


def test_concat_dataframes_missing_index_values(
    feature_one_dummy_df, feature_two_dummy_df
):
    initial_one_df = feature_one_dummy_df
    changed_two_df = feature_two_dummy_df.head(10)
    result_df = _concat_dataframes_on_index(
        [initial_one_df, changed_two_df], index_name="id"
    )
    assert len(result_df) == len(initial_one_df)
    assert result_df.isna().values.sum() > 0


def test_concat_dataframes_missing_index(feature_one_dummy_df, feature_two_dummy_df):
    with pytest.raises(KeyError):
        incorrect_id = "does_not_exist"
        _concat_dataframes_on_index(
            [feature_one_dummy_df, feature_two_dummy_df], index_name=incorrect_id
        )


# _add_suffix_except_col()
def test_add_suffix_correct_column(articles_dummy_df):
    suffix = "_end"
    exception_col = "colour_group_code"

    initial_df_cols = set(articles_dummy_df.columns)
    result_df_cols = _add_suffix_except_col(articles_dummy_df, suffix, exception_col)
    result_df_cols = set(result_df_cols)
    assert initial_df_cols != result_df_cols
    assert len(initial_df_cols) == len(result_df_cols)
    assert (exception_col in initial_df_cols) is True
    assert (exception_col in result_df_cols) is True

    initial_df_cols.remove(exception_col)
    result_df_cols.remove(exception_col)
    assert (initial_df_cols & result_df_cols) == set()


def test_add_suffix_non_existing_except_column(articles_dummy_df):
    suffix = "_end"
    exception_col = "does_not_exist"

    initial_df_cols = set(articles_dummy_df.columns)
    result_df_cols = _add_suffix_except_col(articles_dummy_df, suffix, exception_col)
    result_df_cols = set(result_df_cols)

    assert (initial_df_cols & result_df_cols) == set()
    assert len(initial_df_cols) == len(result_df_cols)
    assert (exception_col in initial_df_cols) is False
    assert (exception_col in result_df_cols) is False


# CHECKING RESULT COLUMNS
class TestResultColumns:
    # articles
    def test_rebuying_articles_isin_cols(self, transactions_dummy_df):
        result_df = _rebuying_articles(transactions_dummy_df)
        assert "perc_rebought" in result_df.columns

    def test_mean_perc_sales_channel_id_isin_cols(self, transactions_dummy_df):
        result_df = _mean_perc_sales_channel_id(transactions_dummy_df)
        assert "perc_article_sales_offline" in result_df.columns

    def test_create_article_features_isin_cols(self, transactions_dummy_df):
        result_df = create_article_features(transactions_dummy_df)
        assert "perc_rebought" in result_df.columns
        assert "perc_article_sales_offline" in result_df.columns

    # customers
    def test_count_of_article_id_per_customer_id_isin_cols(self, transactions_dummy_df):
        result_df = _count_of_article_id_per_customer_id(
            transactions_dummy_df, n_days=None
        )
        assert "count_of_article_per_customer" in result_df.columns

    def test_count_of_product_group_name_per_customer_id_isin_cols(
        self, transactions_dummy_df, articles_dummy_df
    ):
        result_df = _count_of_product_group_name_per_customer_id(
            transactions_dummy_df, articles_dummy_df, n_days=None
        )
        assert "count_of_product_group_name_per_customer" in result_df.columns

    def test_days_since_first_transactions_isin_cols(self, transactions_dummy_df):
        result_df = _days_since_first_transactions(transactions_dummy_df)
        assert "days_since_first_transaction" in result_df.columns

    def test_days_since_last_transactions_isin_cols(self, transactions_dummy_df):
        result_df = _days_since_last_transactions(transactions_dummy_df)
        assert "days_since_last_transaction" in result_df.columns

    def test_average_purchase_span_isin_cols(self, transactions_dummy_df):
        result_df = _average_purchase_span(transactions_dummy_df)
        assert "avg_purchase_span" in result_df.columns

    def test_perc_sales_channel_id_isin_cols(self, transactions_dummy_df):
        result_df = _perc_sales_channel_id(transactions_dummy_df)
        assert "perc_customer_sales_offline" in result_df.columns

    def test_create_customer_features_isin_cols(
        self, transactions_dummy_df, articles_dummy_df
    ):
        result_df = create_customer_features(
            transactions_dummy_df, articles_dummy_df, n_days_list=[None]
        )
        assert "count_of_article_per_customer_all_manual" in result_df.columns
        assert (
            "count_of_product_group_name_per_customer_all_manual" in result_df.columns
        )
        assert "days_since_first_transaction_manual" in result_df.columns
        assert "days_since_last_transaction_manual" in result_df.columns
        assert "avg_purchase_span_manual" in result_df.columns
        assert "perc_customer_sales_offline_manual" in result_df.columns

    # customer x product_group_name features
    def test_count_of_article_id_per_customer_product_group_isin_cols(
        self, transactions_dummy_df, articles_dummy_df
    ):
        result_df = _count_of_article_id_per_customer_product_group(
            transactions_dummy_df, articles_dummy_df
        )
        assert "count_of_article_per_customer_prod_group" in result_df.columns

    def test_create_customer_product_group_features_isin_cols(
        self, transactions_dummy_df, articles_dummy_df
    ):
        result_df = create_customer_product_group_features(
            transactions_dummy_df, articles_dummy_df
        )
        assert "count_of_article_per_customer_prod_group" in result_df.columns
