import pandas as pd

from ga4_mlops.pipelines.prediction.nodes import create_predictions


class TestCreatePredictions:
    def test_classify_on_raw(
        self, abt_sample, raw_scores_sample, calibrated_scores_sample
    ):
        df = create_predictions(
            abt_sample,
            raw_scores_sample,
            calibrated_scores_sample,
            threshold=0.3,
            classify_on_calibrated=False,
        )
        expected_df = pd.DataFrame(
            {
                "i_id": [1, 2, 3, 4, 5],
                "i_info_col": ["2022", "2021", "2022", "2022", "2021"],
                "y_raw_score": [0.99, 0.02, 0.01, 0.98, 0.01],
                "y_calibrated_score": [0.88, 0.42, 0.23, 0.84, 0.22],
                "y_predicted_label": [1, 0, 0, 1, 0],
            }
        )

        assert df.equals(expected_df)

    def test_classify_on_calibrated(
        self, abt_sample, raw_scores_sample, calibrated_scores_sample
    ):
        df = create_predictions(
            abt_sample,
            raw_scores_sample,
            calibrated_scores_sample,
            threshold=0.3,
            classify_on_calibrated=True,
        )
        expected_df = pd.DataFrame(
            {
                "i_id": [1, 2, 3, 4, 5],
                "i_info_col": ["2022", "2021", "2022", "2022", "2021"],
                "y_raw_score": [0.99, 0.02, 0.01, 0.98, 0.01],
                "y_calibrated_score": [0.88, 0.42, 0.23, 0.84, 0.22],
                "y_predicted_label": [1, 1, 0, 1, 0],
            }
        )

        assert df.equals(expected_df)
