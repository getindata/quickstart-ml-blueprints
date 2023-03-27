import logging

import pytest
from autoregressive_forecasting.pipelines.data_processing.pipeline import (
    create_pipeline,
)
from kedro.io.data_catalog import DataCatalog
from kedro.io.memory_dataset import MemoryDataSet
from kedro.runner import SequentialRunner


@pytest.fixture
def seq_runner():
    return SequentialRunner()


@pytest.fixture
def catalog():
    return DataCatalog()


class TestDataProcessingPipeline:
    def test_pipeline(
        self,
        seq_runner,
        catalog,
        raw_data_df,
    ):
        logging.disable("WARNING")
        mapper = {
            "product_id": "unique_id",
            "sale_date": "ds",
            "items_sold": "y",
        }
        min_date_cutoff = None
        max_date_cutoff = None
        # loading up the empty catalog for pipeline inputs.
        catalog.add("input", MemoryDataSet(raw_data_df))
        catalog.add("params:data_processing.column_mapper", MemoryDataSet(mapper))
        catalog.add(
            "params:data_processing.date_cutoffs.min_date",
            MemoryDataSet(min_date_cutoff),
        )
        catalog.add(
            "params:data_processing.date_cutoffs.max_date",
            MemoryDataSet(max_date_cutoff),
        )

        # we want to do an integration test from start to this output
        pipeline = (
            create_pipeline()
            .from_inputs(
                *[
                    "input",
                    "params:data_processing.column_mapper",
                    "params:data_processing.date_cutoffs.min_date",
                    "params:data_processing.date_cutoffs.max_date",
                ]
            )
            .to_outputs("model_input")
        )

        # run the pipeline
        model_input = seq_runner.run(pipeline, catalog)
        df = model_input["model_input"]
        assert set(df.columns) == {"ds", "y", "unique_id"}
