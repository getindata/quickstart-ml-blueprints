"""Project pipelines."""
from autoregressive_forecasting.pipelines import cross_validation as cv
from autoregressive_forecasting.pipelines import data_processing as dp
from autoregressive_forecasting.pipelines import forecasting as f
from autoregressive_forecasting.pipelines import forecasting_with_exogenous_vars as exo
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline()
    forecasting_pipeline = f.create_pipeline()
    cross_validation_pipeline = cv.create_pipeline()
    forecasting_with_exo_pipeline = exo.create_pipeline()

    return {
        "__default__": (
            data_processing_pipeline + forecasting_pipeline + cross_validation_pipeline
        ),
        "data_processing": data_processing_pipeline,
        "forecasting": forecasting_pipeline,
        "cross_validation": cross_validation_pipeline,
        "forecasting_with_exo_vars": forecasting_with_exo_pipeline,
        "end_to_end_forecasting": data_processing_pipeline + forecasting_pipeline,
        "end_to_end_cv": data_processing_pipeline + cross_validation_pipeline,
        "end_to_end_forecasting_with_cv": data_processing_pipeline
        + forecasting_pipeline
        + cross_validation_pipeline,
    }
