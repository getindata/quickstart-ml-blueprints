from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import get_predictions, train_model


def create_pipeline(dataset: str, model: str, comments: str, **kwargs) -> Pipeline:
    """Creates pipeline for graph recommendation models training

    Args:
        dataset (str): dataset name
        model (str): name of gnn model to use
        comments (str): i.e. indication of which subsets we want to create (only_train, train_val, train_val_test)
    """
    namespace = "_".join([dataset, model, comments, "gr"])
    graph_modelling_namespace = "_".join([dataset, model, comments, "grm"])

    pipeline_template = pipeline(
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
                    "params:save_model",
                    "params:seed",
                ],
                outputs="model",
                name="train_model_node",
                tags=["train"],
            ),
            node(
                func=get_predictions,
                inputs=["model", "prediction_set"],
                outputs="predictions",
                name="train_model_node",
                tags=["train"],
            ),
        ]
    )

    main_pipeline = pipeline(
        pipe=pipeline_template,
        inputs={
            "transactions_mapped": f"{dataset}_transactions_mapped",
            "negative_transactions_samples": f"{graph_modelling_namespace}_negative_transactions_samples",
            "train_graphs": f"{graph_modelling_namespace}_train_graphs",
            "val_graphs": f"{graph_modelling_namespace}_val_graphs",
            "test_graphs": f"{graph_modelling_namespace}_test_graphs",
        },
        outputs={
            "model": f"{namespace}_model",
            "predictions": f"{namespace}_predictions",
        },
        namespace=namespace,
    )

    return main_pipeline
