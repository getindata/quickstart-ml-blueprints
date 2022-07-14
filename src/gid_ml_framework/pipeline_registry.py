"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from gid_ml_framework.pipelines import sample_data as sd
from gid_ml_framework.pipelines import image_embeddings as ie
from gid_ml_framework.pipelines import calculate_image_embeddings as cie


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    sample_data_pipeline = sd.create_pipeline()
    image_embeddings_pipeline = ie.create_pipeline()
    calculate_image_embeddings_pipeline = cie.create_pipeline()

    return {
        "__default__": sample_data_pipeline,
        "sd": sample_data_pipeline,
        "ie": image_embeddings_pipeline,
        "cie": calculate_image_embeddings_pipeline,
    }
