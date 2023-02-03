import pandas as pd


class TestGetAndAggregateData:
    def test_type(self, gbq_data_record):
        assert isinstance(gbq_data_record, pd.DataFrame)

    def test_shape(self, gbq_data_record):
        assert gbq_data_record.shape == (1, 3)

    def test_content(self, gbq_data_record):
        expected_data_record = pd.DataFrame(
            {"mobile_brand_name": ["Apple"], "source": ["google"], "platform": ["WEB"]}
        )

        assert gbq_data_record.equals(expected_data_record)
