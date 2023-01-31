import pandas as pd


class TestGetAndAggregateData:
    def test_data_loading(self, gbq_data_record):
        expected_data_record = pd.DataFrame(
            {"mobile_brand_name": ["Apple"], "source": ["google"], "platform": ["WEB"]}
        )

        assert gbq_data_record.equals(expected_data_record)
