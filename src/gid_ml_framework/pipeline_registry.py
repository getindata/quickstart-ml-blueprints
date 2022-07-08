"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from gid_ml_framework.pipelines import sample_data as sd
from gid_ml_framework.pipelines import exploratory_data_analysis as eda


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    sample_data_pipeline = sd.create_pipeline()
    eda_pipeline = eda.create_pipeline()

    return {
        "__default__": sample_data_pipeline,
        "sd": sample_data_pipeline,
        "eda": eda_pipeline
    }
