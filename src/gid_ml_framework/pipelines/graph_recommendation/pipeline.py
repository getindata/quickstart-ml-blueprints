from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import train_model


def create_pipeline(dataset: str, model: str, **kwargs) -> Pipeline:
    """Creates pipeline for graph recommendation models training

    Args:
        dataset (str): dataset name
        model (str): name of gnn model to use
    """
    namespace = "_".join([dataset, model])

    train_pipeline = pipeline(
        [
            node(
                func=train_model,
                inputs=[
                    "train_graphs",
                    "val_graphs",
                    "test_graphs",
                    "transactions_mapped",
                    "negative_transactions_samples",
                    "params:training.model_params",
                    "params:training.train_params",
                ],
                outputs=None,
                name="train_model_node",
                tags=["train"],
            ),
        ]
    )

    main_pipeline = pipeline(
        pipe=train_pipeline,
        inputs={
            "transactions_mapped": f"{dataset}_transactions_mapped",
            "negative_transactions_samples": f"{namespace}_negative_transactions_samples",
            "train_graphs": f"{namespace}_train_graphs",
            "val_graphs": f"{namespace}_val_graphs",
            "test_graphs": f"{namespace}_test_graphs",
        },
        outputs=None,
        namespace=namespace,
    )

    return main_pipeline
