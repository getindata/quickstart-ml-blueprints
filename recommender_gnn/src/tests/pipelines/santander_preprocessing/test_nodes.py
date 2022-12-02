import re

import numpy as np
import pandas as pd
import pytest
from dateutil.parser._parser import ParserError

from recommender_gnn.pipelines.santander_preprocessing.nodes import (
    _impute_age,
    _impute_customer_relation,
    _impute_customer_type,
    _impute_income,
    _impute_income_median,
    _impute_joining_date,
    _impute_new_category,
    _impute_products,
    _impute_seniority,
    _stratify,
    clean_santander,
    filter_santander,
    impute_santander,
    sample_santander,
    split_santander,
)


class TestStratify:
    @pytest.mark.parametrize(
        "customers_limit, expected_result",
        [(3, 4), (30, 30), (50, 50)],
    )
    def test_stratification_shape_given_customers_limit(
        self, bank_dummy_df, customers_limit, expected_result
    ):
        np.random.seed(0)
        assert len(_stratify(bank_dummy_df, customers_limit)) == expected_result

    @pytest.mark.parametrize(
        "customers_limit, expected_result",
        [
            (5, [90037, 173437, 259205, 965820, 1319458]),
            (4, [90037, 259205, 965820, 1319458]),
        ],
    )
    def test_stratification_output_given_customers_limit(
        self, bank_dummy_df, customers_limit, expected_result
    ):
        np.random.seed(0)
        assert all(_stratify(bank_dummy_df, customers_limit) == expected_result)

    def test_stratification_given_not_enough_classes(
        self, bank_small_dummy_df, customers_limit=1
    ):
        with pytest.raises(ValueError):
            _stratify(bank_small_dummy_df, customers_limit)


class TestSampleBank:
    @pytest.mark.parametrize(
        "sample_customer_frac, expected_result",
        [(0, 0), (0.1, 136), (1, 1365)],
    )
    def test_sample_shape_given_customer_frac(
        self, bank_dummy_df, sample_customer_frac, expected_result
    ):
        sample_df = sample_santander(bank_dummy_df, sample_customer_frac)
        assert sample_df.shape[0] == expected_result

    @pytest.mark.parametrize(
        "cutoff_date, expected_result",
        [
            ("2016-05-28", 136),
            ("2015-05-28", 29),
            ("2015-09-28", 60),
            ("2014-09-28", 0),
        ],
    )
    @pytest.mark.parametrize("stratify", [True, False])
    def test_sample_shape_given_cutoff_date(
        self, bank_dummy_df, cutoff_date, expected_result, stratify
    ):
        sample_df = sample_santander(
            bank_dummy_df, cutoff_date=cutoff_date, stratify=stratify
        )
        assert sample_df.shape[0] == expected_result

    @pytest.mark.parametrize(
        "cutoff_date, expected_result",
        [
            ("2016-05-28", "2016-05-28"),
            ("2015-05-28", "2015-05-28"),
            ("2015-09-28", "2015-09-28"),
        ],
    )
    def test_sample_given_valid_cutoff_date(
        self, bank_dummy_df, cutoff_date, expected_result
    ):
        sample_df = sample_santander(bank_dummy_df, cutoff_date=cutoff_date)
        assert sample_df["fecha_dato"].max() <= pd.to_datetime(expected_result)

    def test_sample_given_invalid_cutoff_date(
        self, bank_dummy_df, cutoff_date="wrong format"
    ):
        with pytest.raises(ParserError):
            sample_santander(bank_dummy_df, cutoff_date=cutoff_date)


def test_filter_bank_should_return_expected_columns(bank_dummy_df):
    filtered_df = filter_santander(bank_dummy_df)
    assert filtered_df.shape == (1365, 47)
    assert not set(["tipodom", "cod_prov"]).intersection(filtered_df.columns)


def test_clean_bank_should_fix_string_encoding(bank_dummy_df):
    cleanded_df = clean_santander(bank_dummy_df)
    assert not any(cleanded_df["nomprov"] == "CORU\xc3\x91A, A")


class TestSplitBank:
    def test_split_should_return_same_first_shape(self, bank_dummy_df):
        train_df, test_df = split_santander(bank_dummy_df)
        assert train_df.shape[1] == test_df.shape[1] == bank_dummy_df.shape[1]

    def test_split_should_return_same_columns(self, bank_dummy_df):
        train_df, _ = split_santander(bank_dummy_df)
        assert set(train_df.columns) == set(bank_dummy_df.columns)


class TestImputing:
    @pytest.mark.parametrize(
        "column_name, impute_function",
        [
            ("renta", _impute_income),
            ("age", _impute_age),
            ("tiprel_1mes", _impute_customer_relation),
            ("renta", _impute_income_median),
            ("indrel_1mes", _impute_customer_type),
            ("fecha_alta", _impute_joining_date),
            ("antiguedad", _impute_seniority),
        ],
    )
    def test_single_column_imputing_should_return_no_null_values(
        self, column_name, impute_function, bank_dummy_df
    ):
        imputed_df = impute_function(bank_dummy_df, column_name)
        assert not imputed_df.loc[:, column_name].isnull().values.any()

    def test_impute_products_should_return_no_null_values(self, bank_dummy_df):
        products_pattern = re.compile("ind_+.*ult.*")
        products_cols = list(filter(products_pattern.match, bank_dummy_df.columns))
        imputed_df = _impute_products(bank_dummy_df)
        assert not imputed_df.loc[:, products_cols].isnull().values.any()

    def test_impute_new_category_should_create_new_category_for_null_values(
        self, bank_small_dummy_df
    ):
        missing_column = "conyuemp"
        not_imputed_copy = bank_small_dummy_df.copy()
        imputed_df = _impute_new_category(not_imputed_copy)
        assert imputed_df.loc[
            imputed_df.loc[:, missing_column] == "UNKNOWN"
        ].index.equals(
            bank_small_dummy_df.loc[
                bank_small_dummy_df.loc[:, missing_column].isnull(), missing_column
            ].index
        )

    def test_impute_bank_should_return_no_null_values(self, bank_dummy_df):
        imputed_df = impute_santander(bank_dummy_df)
        dropped_columns = ["tipodom", "cod_prov"]
        imputed_df.drop(dropped_columns, axis=1, inplace=True)
        assert not imputed_df.isnull().any().sum()
