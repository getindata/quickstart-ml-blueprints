from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import generate_graph_dgsr, negative_sample_dgsr, preprocess_dgsr


def create_pipeline(dataset: str, model: str, **kwargs) -> Pipeline:
    namespace = "_".join([dataset, model])

    dgsr_pipeline = pipeline(
        [
            node(
                func=generate_graph_dgsr,
                inputs="transactions",
                outputs="transactions_graph",
                name="generate_graph_dgsr_node",
            ),
            node(
                func=preprocess_dgsr,
                inputs=[
                    "transactions",
                    "transactions_graph",
                ],
                name="preprocess_dgsr_node",
            ),
            node(
                func=negative_sample_dgsr,
                inputs=[
                    "transactions",
                ],
                name="preprocess_dgsr_node",
            ),
        ]
    )

    models_dict = {"dgsr": dgsr_pipeline}

    main_pipeline = pipeline(
        pipe=models_dict.get(model),
        inputs={
            "transactions_train": f"{dataset}_transactions_train",
            "transactions_val": f"{dataset}_transactions_val",
        },
        outputs={
            "transactions_mapped": f"{namespace}_transactions_mapped",
            "users_mapping": f"{namespace}_users_mapping",
            "items_mapping": f"{namespace}_items_mapping",
        },
        namespace=namespace,
    )

    return main_pipeline
