from itertools import chain

from src.gid_ml_framework.pipelines.recommendation_generation.nodes import (
    _ensemble_predictions,
    _weight_blend,
)


# _weight_blend()
def test_weight_blend(predictions_dummy_df):
    prediction_cols = ["prediction_1", "prediction_2"]
    k = 3
    for _, row in predictions_dummy_df.iterrows():
        result = _weight_blend(row, prediction_cols, k)
        assert isinstance(result, str)
        assert len(result.split(" ")) == k


def test_weight_blend_more_k_than_items(predictions_dummy_df):
    prediction_cols = ["prediction_1", "prediction_2"]
    k = 999
    for _, row in predictions_dummy_df.iterrows():
        result = _weight_blend(row, prediction_cols, k)
        assert isinstance(result, str)

        # max possible items
        all_predictions_list_of_lists = [row[col] for col in prediction_cols]
        all_predictions = list(chain(*all_predictions_list_of_lists))
        all_predicitons = set(all_predictions)
        assert len(all_predicitons) == len(result.split(" "))


def test_weight_blend_k_equals_zero(predictions_dummy_df):
    prediction_cols = ["prediction_1", "prediction_2"]
    k = 0
    for _, row in predictions_dummy_df.iterrows():
        result = _weight_blend(row, prediction_cols, k)
        assert isinstance(result, str)
        assert result == ""


# _ensemble_predictions()
def test_ensemble_predictions_cols(predictions_dummy_df):
    result_df = _ensemble_predictions(predictions_dummy_df)
    assert "prediction" in result_df.columns
    assert "customer_id" in result_df.columns


def test_ensemble_predictions_size(predictions_dummy_df):
    initial_df = predictions_dummy_df.copy()
    result_df = _ensemble_predictions(predictions_dummy_df)
    assert len(initial_df) == len(result_df)
    assert len(result_df.columns) == 2


def test_ensemble_predictions_result_is_a_string_of_articles(predictions_dummy_df):
    k = 3
    result_df = _ensemble_predictions(predictions_dummy_df, k)
    for _, row in result_df.iterrows():
        prediction = row["prediction"]
        assert isinstance(prediction, str)
        list_of_articles = prediction.split(" ")
        assert len(list_of_articles) == k
