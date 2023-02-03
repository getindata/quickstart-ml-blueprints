from pandas.api.types import is_categorical_dtype, is_numeric_dtype

from ga4_mlops.pipelines.data_preparation_utils import (
    clean_column_names,
    ensure_column_types,
    extract_column_names,
)


class TestDataPreparationUtils:
    def test_clean_column_names(self, contaminated_column_names_sample):
        df = clean_column_names(contaminated_column_names_sample)
        expected_colnames = [
            "c_first_good_colname",
            "n_second_good_colname",
            "c_first_bad_colname",
            "n_second_bad_colname",
        ]

        assert all(df.columns == expected_colnames)

    def test_ensure_column_types(self, wrong_column_types_sample):
        num_cols = [
            "n_second_good_coltype",
            "n_fourth_good_coltype",
            "n_sixth_good_coltype",
            "n_third_bad_coltype",
            "n_fourth_bad_coltype",
            "n_fifth_bad_coltype",
            "n_seventh_bad_coltype",
        ]
        cat_cols = [
            "c_first_good_coltype",
            "c_third_good_coltype",
            "c_fifth_good_coltype",
            "c_first_bad_coltype",
            "c_second_bad_coltype",
            "c_sixth_bad_coltype",
        ]

        df = ensure_column_types(wrong_column_types_sample, num_cols, cat_cols)

        assert is_numeric_dtype(df["n_second_good_coltype"])
        assert is_numeric_dtype(df["n_fourth_good_coltype"])
        assert is_numeric_dtype(df["n_sixth_good_coltype"])
        assert is_numeric_dtype(df["n_third_bad_coltype"])
        assert is_numeric_dtype(df["n_fourth_bad_coltype"])
        assert is_numeric_dtype(df["n_fifth_bad_coltype"])
        assert is_numeric_dtype(df["n_seventh_bad_coltype"])
        assert is_categorical_dtype(df["c_first_good_coltype"])
        assert is_categorical_dtype(df["c_third_good_coltype"])
        assert is_categorical_dtype(df["c_fifth_good_coltype"])
        assert is_categorical_dtype(df["c_first_bad_coltype"])
        assert is_categorical_dtype(df["c_second_bad_coltype"])
        assert is_categorical_dtype(df["c_sixth_bad_coltype"])

    def test_extract_column_names(self, column_names_sample):
        info_cols, num_cols, cat_cols, target_col = extract_column_names(
            column_names_sample
        )
        expected_info_cols = ["i_info_col_1", "i_info_col_2"]
        expected_num_cols = ["n_num_col_1", "n_num_col_2"]
        expected_cat_cols = ["c_cat_col"]
        expected_target_col = "y_target_col"

        assert info_cols == expected_info_cols
        assert num_cols == expected_num_cols
        assert cat_cols == expected_cat_cols
        assert target_col == expected_target_col
