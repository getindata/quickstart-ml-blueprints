from autoregressive_forecasting.pipelines.data_processing.nodes import _sum_sales
from autoregressive_forecasting.pipelines.data_processing.nodes import preprocess_data


class TestDataProcessing:
    # show basic functionality
    def test_preprocess_data(self, raw_data_df):
        correct_mapper = {
            "product_id": "unique_id",
            "sale_date": "ds",
            "items_sold": "y",
        }
        min_date_cutoff = None
        max_date_cutoff = None
        df_output = preprocess_data(
            raw_data_df, correct_mapper, min_date_cutoff, max_date_cutoff
        )
        expected_cols_set = {"unique_id", "ds", "y"}
        assert set(df_output.columns) == expected_cols_set
        assert df_output.shape == raw_data_df.shape

    def test_sum_sales(self, model_input_df):
        df = model_input_df
        df["new_column"] = "123"
        df_summed = _sum_sales(df)

        assert "new_column" not in set(df_summed.columns)
        assert set(df_summed.columns) == {"unique_id", "ds", "y"}
