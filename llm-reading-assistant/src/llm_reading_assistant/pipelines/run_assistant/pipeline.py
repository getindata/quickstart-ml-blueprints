"""
This is a boilerplate pipeline 'run_assistant'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import complete_request


def create_pipeline(**kwargs) -> Pipeline:
    main_pipeline = pipeline(
        [
            node(
                func=complete_request,
                name="execute_prompt_node",
                inputs=[
                    "params:api",
                    "params:mode",
                    "params:input_text",
                    "params:instructions",
                    "params:max_tokens",
                    "params:model",
                ],
                outputs="answer",
            ),
        ]
    )

    return main_pipeline
