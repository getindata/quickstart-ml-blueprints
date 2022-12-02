import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture

from recommender_gnn.pipelines.santander_to_act.nodes import (
    _identify_newly_added,
    santander_to_articles,
    santander_to_customers,
    santander_to_transactions,
)


@pytest.mark.parametrize(
    "subset_df",
    [lazy_fixture("preprocessed_train_df"), lazy_fixture("preprocessed_val_df")],
)
def test_bank_to_articles_should_return_valid_unique_articles(subset_df):
    articles = santander_to_articles(subset_df)
    article_id_column = "article_id"
    assert articles.shape == (24, 1)
    assert articles.loc[:, article_id_column].is_unique


class TestBankToCustomers:
    customer_id_column = "customer_id"

    def test_given_merge_type_last_should_return_valid_unique_customers(
        self, preprocessed_train_df, preprocessed_val_df
    ):
        merge_type = "last"
        customers = santander_to_customers(
            preprocessed_train_df, preprocessed_val_df, merge_type
        )
        assert customers.shape == (2224, 46)
        assert customers.loc[:, self.customer_id_column].is_unique

    def test_given_no_merge_type_should_return_valid_unique_customers(
        self, preprocessed_train_df, preprocessed_val_df
    ):
        customers = santander_to_customers(preprocessed_train_df, preprocessed_val_df)
        assert customers.shape == (2224, 1)
        assert customers.loc[:, self.customer_id_column].is_unique


class TestIdentifyNewlyAdded:
    @pytest.mark.parametrize(
        "subset_df, expected_result",
        [
            (lazy_fixture("preprocessed_train_df"), (1292, 47)),
            (lazy_fixture("preprocessed_val_df"), (957, 47)),
        ],
    )
    def test_identify_newly_added_shape(self, subset_df, expected_result):
        newly_added_df, _ = _identify_newly_added(subset_df, pd.DataFrame({}))
        assert newly_added_df.shape == expected_result

    @pytest.mark.parametrize(
        "subset_df, expected_result",
        [
            (lazy_fixture("preprocessed_train_df"), 31007),
            (lazy_fixture("preprocessed_val_df"), 22968),
        ],
    )
    def test_identify_newly_added_values(self, subset_df, expected_result):
        newly_added_df, _ = _identify_newly_added(subset_df, pd.DataFrame({}))
        print(newly_added_df)
        assert newly_added_df.eq("Maintained").sum().sum() == expected_result


@pytest.mark.parametrize(
    "train_df, val_df, expected_result",
    [
        (
            lazy_fixture("preprocessed_train_df"),
            lazy_fixture("preprocessed_val_df"),
            (3, 3),
        ),
        (
            lazy_fixture("preprocessed_train_bigger_df"),
            lazy_fixture("preprocessed_val_df"),
            (104, 3),
        ),
    ],
)
def test_bank_to_transaction_should_return_df_of_expected_shape(
    train_df, val_df, expected_result
):
    transactions_train, transactions_val = santander_to_transactions(train_df, val_df)
    transactions = pd.concat([transactions_train, transactions_val])
    assert transactions.shape == expected_result
