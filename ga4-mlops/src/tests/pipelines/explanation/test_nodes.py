import pandas as pd

from ga4_mlops.pipelines.explanation.nodes import sample_data


class TestSampleData:
    def test_sample_data(self, explanation_sample):
        df = sample_data(explanation_sample, n_obs=8, seed=42)
        expected_df = pd.DataFrame(
            {
                "i_id": [14, 16, 5, 18, 10, 13, 1, 4],
                "y_target": [0, 0, 0, 0, 0, 0, 1, 1],
            }
        )

        assert df.equals(expected_df)
