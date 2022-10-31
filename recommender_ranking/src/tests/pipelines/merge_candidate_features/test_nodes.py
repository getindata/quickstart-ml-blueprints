import pandas as pd
import pytest

from src.gid_ml_framework.pipelines.merge_candidate_features.nodes import (
    _cast_as_category,
    _fill_na_cast_to_int,
    add_article_features,
    add_customer_features,
    add_dict_features,
    add_label,
)


class TestFillNACastToInt:
    def test_fill_na_correct_int(self, customers_dummy_df):
        fill_na_int = 0
        initial_df = customers_dummy_df.copy()
        result_df = _fill_na_cast_to_int(
            customers_dummy_df, regex_pattern="FN", fill_na_value=fill_na_int
        )
        assert result_df.loc[:, "FN"].isna().sum() == 0
        assert initial_df.loc[:, "FN"].isna().sum() > 0

    def test_fill_na_correct_str(self, customers_dummy_df):
        fill_na_str = "0"
        initial_df = customers_dummy_df.copy()
        result_df = _fill_na_cast_to_int(
            customers_dummy_df, regex_pattern="FN", fill_na_value=fill_na_str
        )
        assert result_df.loc[:, "FN"].isna().sum() == 0
        assert initial_df.loc[:, "FN"].isna().sum() > 0

    def test_not_existing_column(self, customers_dummy_df):
        wrong_regex_pattern = "does_not_exist"
        initial_df = customers_dummy_df.copy()
        result_df = _fill_na_cast_to_int(
            customers_dummy_df, regex_pattern=wrong_regex_pattern, fill_na_value=0
        )
        pd.testing.assert_frame_equal(initial_df, result_df)

    def test_fill_with_wrong_string(self, customers_dummy_df):
        wrong_fill_value = "abcdef"
        with pytest.raises(ValueError) as exc_info:
            _fill_na_cast_to_int(
                customers_dummy_df, regex_pattern="FN", fill_na_value=wrong_fill_value
            )
            assert "invalid literal for int() with base 10:" in exc_info.value.args[0]

    def test_fill_column_without_na(self, customers_dummy_df):
        initial_df = customers_dummy_df.copy()
        initial_df["age"] = initial_df["age"].astype(int)
        result_df = _fill_na_cast_to_int(
            customers_dummy_df, regex_pattern="age", fill_na_value=0
        )
        pd.testing.assert_frame_equal(initial_df, result_df)


class TestCastAsCategory:
    def test_single_object_conversion(self, customers_dummy_df):
        # is this the correct way of testing types in pandas?
        conversion_col = "fashion_news_frequency"
        initial_df = customers_dummy_df.copy()
        result_df = _cast_as_category(customers_dummy_df, conversion_col)
        # assert result_df.dtypes[conversion_col].name=='category'
        # assert initial_df.dtypes[conversion_col].name!='category'
        assert (
            issubclass(
                result_df[conversion_col].dtype.type,
                pd.core.dtypes.dtypes.CategoricalDtypeType,
            )
            is True
        )
        assert (
            issubclass(
                initial_df[conversion_col].dtype.type,
                pd.core.dtypes.dtypes.CategoricalDtypeType,
            )
            is False
        )

    def test_multiple_object_conversion(self, customers_dummy_df):
        conversion_cols = ["fashion_news_frequency", "club_member_status"]
        initial_df = customers_dummy_df.copy()
        result_df = _cast_as_category(customers_dummy_df, conversion_cols)
        assert (
            issubclass(
                result_df[conversion_cols[0]].dtype.type,
                pd.core.dtypes.dtypes.CategoricalDtypeType,
            )
            is True
        )
        assert (
            issubclass(
                initial_df[conversion_cols[0]].dtype.type,
                pd.core.dtypes.dtypes.CategoricalDtypeType,
            )
            is False
        )
        assert (
            issubclass(
                result_df[conversion_cols[1]].dtype.type,
                pd.core.dtypes.dtypes.CategoricalDtypeType,
            )
            is True
        )
        assert (
            issubclass(
                initial_df[conversion_cols[1]].dtype.type,
                pd.core.dtypes.dtypes.CategoricalDtypeType,
            )
            is False
        )

    def test_integer_to_category_conversion(self, customers_dummy_df):
        conversion_col = "age"
        initial_df = customers_dummy_df.copy()
        result_df = _cast_as_category(customers_dummy_df, conversion_col)
        assert (
            issubclass(
                result_df[conversion_col].dtype.type,
                pd.core.dtypes.dtypes.CategoricalDtypeType,
            )
            is True
        )
        assert (
            issubclass(
                initial_df[conversion_col].dtype.type,
                pd.core.dtypes.dtypes.CategoricalDtypeType,
            )
            is False
        )

    def test_not_existing_column(self, customers_dummy_df):
        conversion_col = "does_not_exist"
        with pytest.raises(KeyError):
            _cast_as_category(customers_dummy_df, conversion_col)

    def test_empty_list(self, customers_dummy_df):
        conversion_cols = []
        initial_df = customers_dummy_df.copy()
        result_df = _cast_as_category(customers_dummy_df, conversion_cols)
        pd.testing.assert_frame_equal(initial_df, result_df)


class TestAddLabel:
    def test_add_label(self, candidates_dummy_df, transactions_dummy_df):
        result_df = add_label(candidates_dummy_df, transactions_dummy_df)
        assert set(list(result_df["label"].unique())) == set([0, 1])
        assert result_df["label"].isna().sum() == 0

    def test_none_transactions(self, candidates_dummy_df):
        "It's actually never used, since there are no OPTIONAL NODES in kedro"
        initial_df = candidates_dummy_df.copy()
        result_df = add_label(candidates_dummy_df, None)
        pd.testing.assert_frame_equal(initial_df, result_df)

    def test_merge_non_overlapping_records(
        self, candidates_dummy_df, transactions_dummy_df
    ):
        initial_transactions_df = transactions_dummy_df
        initial_transactions_df["customer_id"] = range(20)
        initial_transactions_df["customer_id"] = initial_transactions_df[
            "customer_id"
        ].astype(str)
        initial_transactions_df["article_id"] = range(20, 40)
        initial_transactions_df["article_id"] = initial_transactions_df[
            "article_id"
        ].astype(str)

        result_df = add_label(candidates_dummy_df, transactions_dummy_df)
        assert set(list(result_df["label"].unique())) == set([0])
        assert result_df["label"].isna().sum() == 0


class TestAddFeatures:
    """
    TODO: These functions look alike, maybe refactor, and just use one function 3 times?
    """

    def test_add_article_features(self, candidates_dummy_df, articles_dummy_df):
        regex_pattern = "does_not_exist"
        automated_article_features = articles_dummy_df.copy()[
            ["article_id", "product_code", "prod_name"]
        ]
        manual_article_features = articles_dummy_df.copy()[
            ["article_id", "product_type_name", "product_group_name"]
        ]
        result_df = add_article_features(
            candidates_dummy_df,
            automated_article_features,
            manual_article_features,
            regex_pattern=regex_pattern,
        )
        merged_cols = set(
            ["product_code", "prod_name", "product_type_name", "product_group_name"]
        )
        result_cols = set(result_df.columns.to_list())
        assert result_cols.intersection(merged_cols) == merged_cols

    def test_add_customer_features(self, candidates_dummy_df, customers_dummy_df):
        regex_pattern = "does_not_exist"
        automated_customers_features = customers_dummy_df.copy()[
            ["customer_id", "FN", "club_member_status"]
        ]
        manual_customers_features = customers_dummy_df.copy()[
            ["customer_id", "Active", "fashion_news_frequency", "age"]
        ]
        result_df = add_customer_features(
            candidates_dummy_df,
            automated_customers_features,
            manual_customers_features,
            regex_pattern=regex_pattern,
        )
        merged_cols = set(
            ["FN", "club_member_status", "Active", "fashion_news_frequency", "age"]
        )
        result_cols = set(result_df.columns.to_list())
        assert result_cols.intersection(merged_cols) == merged_cols

    def test_add_dict_features(
        self, candidates_dummy_df, articles_dummy_df, customers_dummy_df
    ):
        initial_candidates_df = candidates_dummy_df
        category_cols = ["fashion_news_frequency", "product_group_name"]
        drop_cols = ["graphical_appearance_no", "postal_code"]
        result_df = add_dict_features(
            candidates_dummy_df,
            articles_dummy_df,
            customers_dummy_df,
            category_cols=category_cols,
            drop_cols=drop_cols,
        )
        assert len(initial_candidates_df) == len(result_df)

    def test_add_dict_features_drop_cols(
        self, candidates_dummy_df, articles_dummy_df, customers_dummy_df
    ):
        category_cols = ["fashion_news_frequency", "product_group_name"]
        drop_cols = ["graphical_appearance_no", "postal_code"]
        result_df = add_dict_features(
            candidates_dummy_df,
            articles_dummy_df,
            customers_dummy_df,
            category_cols=category_cols,
            drop_cols=drop_cols,
        )
        result_cols = set(result_df.columns.to_list())
        assert ("graphical_appearance_no" in result_cols) is False
        assert ("postal_code" in result_cols) is False
        assert ("graphical_appearance_no" in articles_dummy_df.columns) is True
        assert ("postal_code" in customers_dummy_df.columns) is True

    def test_add_dict_features_no_drop_cols(
        self, candidates_dummy_df, articles_dummy_df, customers_dummy_df
    ):
        initial_candidates_df = candidates_dummy_df
        category_cols = ["fashion_news_frequency", "product_group_name"]
        drop_cols = []
        result_df = add_dict_features(
            candidates_dummy_df,
            articles_dummy_df,
            customers_dummy_df,
            category_cols=category_cols,
            drop_cols=drop_cols,
        )
        expected_len_cols = (
            len(initial_candidates_df.columns)
            + len(articles_dummy_df.columns)
            - 1
            + len(customers_dummy_df.columns)
            - 1
        )
        result_len_cols = len(result_df.columns)
        assert expected_len_cols == result_len_cols
