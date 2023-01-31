import pandas as pd

from ga4_mlops.pipelines.feature_engineering.nodes import engineer_features


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
    def test_fit_and_apply_imputers(self):
        pass


class TestEncoding:
    def test_fit_and_apply_encoders(self):
        pass
