from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import get_predictions, test_model, train_model


def create_pipeline(
    dataset: str, model: str, comments: str = None, **kwargs
) -> Pipeline:
    """Creates pipeline for graph recommendation models training

    Args:
        dataset (str): dataset name
        model (str): name of gnn model to use
        comments (str): i.e. indication of which subsets we want to create (only_train, train_val, train_val_test)
    """
    namespace_list = [x for x in [dataset, model, comments] if x is not None]
    namespace = "_".join(["graph_recommendation"] + namespace_list)
    grm_namespace = "_".join(["graph_recommendation_modelling"] + namespace_list)
    grp_namespace = "_".join(
        ["graph_recommendation_preprocessing"] + namespace_list[::2]
    )

    pipeline_template = pipeline(
        [
            node(
                func=train_model,
                inputs=[
                    "train_graphs",
                    "val_graphs",
                    "transactions_mapped",
                    "negative_transactions_samples",
                    "params:training.model_params",
                    "params:training.train_params",
                    "params:save_model",
                    "params:seed",
                ],
                outputs=["model", "data_stats"],
                name="train_model_node",
                tags=["gpu_tag"],
            ),
            node(
                func=test_model,
                inputs=[
                    "test_graphs",
                    "model",
                    "negative_transactions_samples",
                    "params:training.train_params",
                    "data_stats",
                ],
                outputs=None,
                name="test_model_node",
                tags=["gpu_tag"],
            ),
            node(
                func=get_predictions,
                inputs=["prediction_graphs", "model", "params:training.train_params"],
                outputs="predictions",
                name="get_predictions_node",
                tags=["gpu_tag"],
            ),
        ]
    )

    main_pipeline = pipeline(
        pipe=pipeline_template,
        inputs={
            "transactions_mapped": f"{grp_namespace}.transactions_mapped",
            "negative_transactions_samples": f"{grm_namespace}.negative_transactions_samples",
            "train_graphs": f"{grm_namespace}.train_graphs",
            "val_graphs": f"{grm_namespace}.val_graphs",
            "test_graphs": f"{grm_namespace}.test_graphs",
            "prediction_graphs": f"{grm_namespace}.predict_graphs",
        },
        namespace=namespace,
    )

    return main_pipeline
