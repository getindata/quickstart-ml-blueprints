from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import generate_graph_dgsr, negative_sample_dgsr, preprocess_dgsr


def create_pipeline(dataset: str, model: str, **kwargs) -> Pipeline:
    """Creates pipeline for graph

    Args:
        dataset (str): dataset name
        model (str): name of gnn model to use

    Returns:
        Tuple: tuple of dataframes including original dataframe with mapping applied and mappings for users and items
    """
    namespace = "_".join([dataset, model])

    dgsr_pipeline = pipeline(
        [
            node(
                func=generate_graph_dgsr,
                inputs="transactions_mapped",
                outputs="transactions_graph",
                name="generate_graph_dgsr_node",
                tags=["preprocess", "all"],
            ),
            node(
                func=preprocess_dgsr,
                inputs=[
                    "transactions_mapped",
                    "transactions_graph",
                ],
                outputs=[
                    "train_graphs",
                    "val_graphs",
                    "test_graphs",
                ],
                name="preprocess_dgsr_node",
                tags=["preprocess", "all"],
            ),
            node(
                func=negative_sample_dgsr,
                inputs="transactions_mapped",
                outputs="negative_transactions_samples",
                name="negative_sample_dgsr_node",
                tags=["preprocess", "all"],
            ),
        ]
    )

    models_dict = {"dgsr": dgsr_pipeline}

    main_pipeline = pipeline(
        pipe=models_dict.get(model),
        inputs={"transactions_mapped": f"{dataset}_transactions_mapped"},
        outputs={
            "transactions_graph": f"{namespace}_transactions_graph",
            "negative_transactions_samples": f"{namespace}_negative_transactions_samples",
            "train_graphs": f"{namespace}_train_graphs",
            "val_graphs": f"{namespace}_val_graphs",
            "test_graphs": f"{namespace}_test_graphs",
        },
        namespace=namespace,
    )

    return main_pipeline
