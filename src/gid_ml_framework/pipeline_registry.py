"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from gid_ml_framework.pipelines import sample_data as sd
from gid_ml_framework.pipelines import exploratory_data_analysis as eda
from gid_ml_framework.pipelines import image_embeddings as ie
from gid_ml_framework.pipelines import calculate_image_embeddings as cie
from gid_ml_framework.pipelines import text_embeddings as te
from gid_ml_framework.pipelines import image_resizer as ir
from gid_ml_framework.pipelines import candidate_generation as cg
from gid_ml_framework.pipelines import train_val_split as tvs
from gid_ml_framework.pipelines import candidate_generation_validation as cgv


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    sample_data_pipeline = sd.create_pipeline()
    eda_pipeline = eda.create_pipeline()
    image_embeddings_pipeline = ie.create_pipeline()
    calculate_image_embeddings_pipeline = cie.create_pipeline()
    text_embeddings_pipeline = te.create_pipeline()
    image_resizer_pipeline = ir.create_pipeline()
    candidate_generation_pipeline = cg.create_pipeline()
    train_val_split_pipeline = tvs.create_pipeline()
    candidate_generation_validation_pipeline = cgv.create_pipeline()

    return {
        "__default__": sample_data_pipeline,
        "sd": sample_data_pipeline,
        "eda": eda_pipeline,
        "ie": image_embeddings_pipeline,
        "cie": calculate_image_embeddings_pipeline,
        "te": text_embeddings_pipeline,
        "ir": image_resizer_pipeline,
        "cg": candidate_generation_pipeline,
        "tvs": train_val_split_pipeline,
        "cgv": candidate_generation_validation_pipeline,
        "candidate_generation": (train_val_split_pipeline + candidate_generation_pipeline + candidate_generation_validation_pipeline)
    }
