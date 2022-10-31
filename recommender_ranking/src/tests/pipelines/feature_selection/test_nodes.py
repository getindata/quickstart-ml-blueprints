import pandas as pd
import pytest

from src.gid_ml_framework.pipelines.feature_selection.nodes import (
    _remove_correlated_features,
    feature_selection,
)


# _remove_correlated_features()
def test_corr_threshold_equal_one(dummy_integers_df):
    initial_df = dummy_integers_df
    result_df = _remove_correlated_features(dummy_integers_df, corr_threshold=1.0)
    pd.testing.assert_frame_equal(initial_df, result_df)


def test_corr_threshold_close_to_zero(dummy_integers_df):
    result_df = _remove_correlated_features(dummy_integers_df, corr_threshold=0.001)
    assert len(result_df.columns) == 1
    assert result_df["A"].empty is False


def test_corr_threshold_equal_to_zero(dummy_integers_df):
    with pytest.raises(ValueError) as exc_info:
        _remove_correlated_features(dummy_integers_df, corr_threshold=0)
    assert exc_info.type is ValueError
    assert (
        exc_info.value.args[0]
        == "Correlation threshold should be a float value between (0, 1>"
    )


def test_corr_threshold_above_one(dummy_integers_df):
    with pytest.raises(ValueError) as exc_info:
        _remove_correlated_features(dummy_integers_df, corr_threshold=1.01)
    assert exc_info.type is ValueError
    assert (
        exc_info.value.args[0]
        == "Correlation threshold should be a float value between (0, 1>"
    )


def test_correct_corr_threshold(dummy_integers_df):
    initial_df = dummy_integers_df
    result_df = _remove_correlated_features(dummy_integers_df, corr_threshold=0.4)
    assert len(initial_df.columns) > len(result_df.columns)
    assert len(initial_df) == len(result_df)


def test_string_dataframe(dummy_string_df):
    with pytest.raises(ValueError) as exc_info:
        _remove_correlated_features(dummy_string_df)
    assert exc_info.type is ValueError
    assert (
        exc_info.value.args[0]
        == "DataFrame contains only non-numeric data, cannot compute correlations"
    )


def test_integers_and_string_dataframe(dummy_integers_df):
    integer_df = dummy_integers_df.copy()
    string_df = dummy_integers_df.copy()
    string_df = string_df.assign(F=lambda x: "some_string_value")
    result_integer_df = _remove_correlated_features(integer_df, corr_threshold=0.4)
    result_string_df = _remove_correlated_features(string_df, corr_threshold=0.4)
    assert len(result_string_df.columns) == len(result_integer_df.columns) + 1
    assert result_string_df.loc[:, "F"].empty is False


# feature_selection()
def test_feature_selection_is_false(dummy_integers_df):
    initial_cols = set(dummy_integers_df.columns)
    cols = feature_selection(
        dummy_integers_df,
        selection_params={"pct_null_threshold": 0.8, "corr_threshold": 0.8},
        feature_selection=False,
    )
    assert cols == initial_cols
