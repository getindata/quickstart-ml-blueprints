import pytest
from kedro.extras.datasets.pandas import GBQQueryDataSet


@pytest.fixture
def gbq_data_record():
    sql = """
        SELECT device.mobile_brand_name, traffic_source.source, platform
        FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210131`
        WHERE event_timestamp = 1612069510766593 AND event_name = "page_view"
    """
    gbq_data_set = GBQQueryDataSet(sql)
    df = gbq_data_set.load()

    return df
