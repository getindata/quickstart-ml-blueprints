from recommender_gnn.pipelines.kaggle_submission.nodes import (
    _borda_sort,
    _impute_missing_predictions,
    generate_submission,
)


class TestGenerateSubmission:
    user_column = "user_id"
    item_column = "item_id"

    @classmethod
    def get_item_columns(cls, df):
        item_columns = [col for col in df.columns if col != cls.user_column]
        item_columns = sorted(item_columns, key=int)
        return item_columns

    def test_borda_sort(self, small_predictions_custom):
        item_columns = self.get_item_columns(small_predictions_custom)
        rankings = small_predictions_custom.loc[:, item_columns].values.tolist()
        result = _borda_sort(rankings)
        assert result == [1, 0, 2, 3]

    def test_impute_missing_predictions(
        self, small_predictions_custom, small_users_custom
    ):
        item_columns = self.get_item_columns(small_predictions_custom)
        imputed_predcitions = _impute_missing_predictions(
            small_predictions_custom.iloc[
                1:,
            ],
            small_users_custom,
            self.user_column,
            item_columns,
            self.item_column,
            self.user_column,
        )
        assert imputed_predcitions.shape == (1, 2)

    def test_generate_submission_shape(
        self,
        small_predictions_custom,
        small_users_custom,
        small_items_mapping_custom,
        small_users_mapping_custom,
    ):
        submission_df = generate_submission(
            small_predictions_custom.iloc[
                1:,
            ],
            small_users_custom,
            small_users_mapping_custom,
            small_items_mapping_custom,
            None,
            self.item_column,
            self.user_column,
            self.user_column,
            None,
        )
        assert submission_df.shape == (6, 2)
