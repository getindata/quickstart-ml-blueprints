"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from llm_reading_assistant.pipelines import run_assistant


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    run_assistant_pipeline = run_assistant.create_pipeline()

    return {
        "__default__": run_assistant_pipeline,
        "run_assistant_pipeline": run_assistant_pipeline,
    }
