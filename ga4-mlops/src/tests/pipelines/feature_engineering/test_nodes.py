import numpy as np
import pandas as pd

from ga4_mlops.pipelines.feature_engineering.nodes import (
    apply_encoders,
    apply_imputers,
    engineer_features,
    exclude_features,
    fit_encoders,
    fit_imputers,
)


class TestEngineerFeatures:
    def test_transformation(self, engineer_features_sample):
        df = engineer_features(engineer_features_sample)
        expected_df = pd.DataFrame(
            {
                "c_visit_start_hour": [0, 7, 11],
                "c_weekday": [1, 3, 1],
                "i_full_visitor_id": [
                    3033910.355860057,
                    81793309.0616803,
                    4293031.296243032,
                ],
                "i_visit_start_time": [
                    1611619614341157.0,
                    1611820704267587.0,
                    1611661585573344.0,
                ],
            }
        )

        assert df.sort_index(axis=1).equals(expected_df.sort_index(axis=1))


class TestImputation:
    def test_fit_and_apply_imputers(self, imputation_sample):
        imputation_strategies = {
            "mean": [
                "n_num_col_missing_for_mean"
            ],  # for numerical: replace NULLs with column mean
            "zero": [
                "n_num_col_missing_for_zero"
            ],  # for numerical: replace NULLs with zeros
            "mostfreq": [
                "c_cat_col_for_mostfreq"
            ],  # for categorical: replace NULLs with most frequent value
            "unknown": [
                "c_cat_col_for_unknown"
            ],  # for categorical: replace NULLs with UNKNOWN token
        }
        imputers = fit_imputers(imputation_sample, imputation_strategies)
        df = apply_imputers(imputation_sample, imputers)
        expected_df = pd.DataFrame(
            {
                "i_id_col": [1, 2, 3, 4, 5],
                "n_num_col_complete": [4.0, 2.0, 6.0, 1.0, 0.0],
                "n_num_col_missing_not_listed": [4.0, np.nan, 2.1, 3.2, np.nan],
                "n_num_col_missing_for_mean": [2.0, 5.0, 5.0, 5.0, 8.0],
                "n_num_col_missing_for_zero": [0.0, 9.3, -1.2, 0.0, 0.0],
                "c_cat_col_for_mostfreq": ["cat1", "cat1", "cat2", "cat1", "cat1"],
                "c_cat_col_for_unknown": ["cat1", "cat2", "UNKNOWN", "cat3", "cat2"],
            }
        )

        assert df.equals(expected_df)


class TestEncoding:
    def test_fit_and_apply_encoders(self, encoding_sample):
        encoder_types = {
            "binary": ["c_cat_col_for_binary"],  # for binary variables
            "onehot": ["c_cat_col_for_onehot"],  # one-hot encoding
            "ordinal": [
                "c_cat_col_for_ordinal"
            ],  # integer encoding (ordinal, but order does not matter)
        }
        encoders = fit_encoders(encoding_sample, encoder_types)
        df = apply_encoders(encoding_sample, encoders)
        expected_df = pd.DataFrame(
            {
                "i_id_col": [1, 2, 3, 4, 5],
                "c_cat_col_for_binary_0": [0, 0, 1, 0, 1],
                "c_cat_col_for_binary_1": [1, 1, 0, 1, 0],
                "c_cat_col_for_onehot_cat1": [1, 0, 0, 0, 0],
                "c_cat_col_for_onehot_cat2": [0, 1, 0, 0, 1],
                "c_cat_col_for_onehot_cat3": [0, 0, 1, 1, 0],
                "c_cat_col_for_ordinal": [1, 2, 3, 1, 2],
            }
        )

        assert df.equals(expected_df)


class TestExcludeFeatures:
    def test_exclude_features(self, exclusion_sample):
        features_to_exclude = ["c_first_col_to_exclude", "c_second_col_to_exclude"]
        df = exclude_features(exclusion_sample, features_to_exclude)
        expected_df = pd.DataFrame(
            {
                "i_id_col": [1, 2, 3, 4, 5],
                "n_col_to_use": [3.2, 2.1, 9.8, 0.3, 3.1],
                "c_col_to_use_0": [0, 1, 0, 0, 0],
                "c_col_to_use_1": [1, 0, 1, 1, 1],
            }
        )

        assert df.equals(expected_df)
