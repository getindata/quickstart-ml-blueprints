import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture

from recommender_gnn.pipelines.otto_to_act.nodes import (
    extract_articles,
    extract_customers,
    extract_transactions,
)


class TestExtractTransactions:
    columns = ["customer_id", "article_id", "timestamp"]

    @pytest.mark.parametrize(
        "dummy_df, expected_result",
        [
            (lazy_fixture("otto_dummy_df"), (6928, 3)),
            (lazy_fixture("otto_small_dummy_df"), (2068, 3)),
        ],
    )
    def test_extract_transactions_shape_and_columns(self, dummy_df, expected_result):
        print(dummy_df)
        transactions = extract_transactions(dummy_df)
        print(transactions)
        assert transactions.shape == expected_result
        assert transactions.columns.tolist() == self.columns

    def test_extract_transactions_given_wrong_columns_name_should_raise_error(
        self, otto_dummy_df
    ):
        with pytest.raises(KeyError):
            extract_transactions(
                otto_dummy_df, original_article_col="wrong_column_name"
            )

    def test_extract_transactions_given_empty_df_should_raise_error(self):
        with pytest.raises(KeyError):
            extract_transactions(pd.DataFrame({}))


class TestExtractArticles:
    article_id_column = "aid"

    @pytest.mark.parametrize(
        "dummy_df, expected_result",
        [
            (lazy_fixture("otto_dummy_df"), (6394, 1)),
            (lazy_fixture("otto_small_dummy_df"), (1878, 1)),
        ],
    )
    def test_extract_articles_should_return_valid_unique_articles(
        self, dummy_df, expected_result
    ):
        articles = extract_articles(dummy_df, self.article_id_column)
        assert articles.shape == expected_result
        assert articles.loc[:, self.article_id_column].is_unique

    def test_extract_articles_given_empty_df_should_raise_error(self):
        with pytest.raises(KeyError):
            extract_articles(pd.DataFrame({}), self.article_id_column)


class TestExtractCustomers:
    customer_id_column = "session"

    @pytest.mark.parametrize(
        "dummy_df, expected_result",
        [
            (lazy_fixture("otto_dummy_df"), (6864, 1)),
            (lazy_fixture("otto_small_dummy_df"), (686, 1)),
        ],
    )
    def test_extract_customers_should_return_valid_unique_customers(
        self, dummy_df, expected_result
    ):
        customers = extract_customers(dummy_df, self.customer_id_column)
        assert customers.shape == expected_result
        assert customers.loc[:, self.customer_id_column].is_unique

    def test_extract_customers_given_empty_df_should_raise_error(self):
        with pytest.raises(KeyError):
            extract_customers(pd.DataFrame({}), self.customer_id_column)
