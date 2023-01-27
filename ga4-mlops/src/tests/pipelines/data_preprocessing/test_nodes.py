import pandas as pd


class TestGetAndAggregateData:
    def test_data_loading(self, gbq_data_record):
        data_record = gbq_data_record

        expected_data_record = pd.DataFrame(
            {"mobile_brand_name": ["Apple"], "source": ["google"], "platform": ["WEB"]}
        )

        assert data_record.equals(expected_data_record)
