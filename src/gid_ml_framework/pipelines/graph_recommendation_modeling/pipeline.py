from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import generate_graph_dgsr, preprocess_dgsr, sample_negatives_dgsr


def create_pipeline(dataset: str, model: str, comments: str, **kwargs) -> Pipeline:
    """Creates pipeline for graph data modeling for given GNN model

    Args:
        dataset (str): dataset name
        model (str): name of gnn model to use
        comments (str): i.e. indication of which subsets we want to create graphs
    """
    namespace = "_".join([dataset, model, comments, "grm"])

    dgsr_pipeline = pipeline(
        [
            node(
                func=generate_graph_dgsr,
                inputs="transactions_mapped",
                outputs="transactions_graph",
                name="generate_graph_node",
                tags=["preprocess"],
            ),
            node(
                func=preprocess_dgsr,
                inputs=[
                    "transactions_mapped",
                    "transactions_graph",
                    "params:preprocess.item_max_length",
                    "params:preprocess.user_max_length",
                    "params:preprocess.k_hop",
                    "params:preprocess.val_flag",
                    "params:preprocess.test_flag",
                    "params:preprocess.predict_flag",
                ],
                outputs=[
                    "train_graphs",
                    "val_graphs",
                    "test_graphs",
                    "predict_graphs",
                ],
                name="preprocess_node",
                tags=["preprocess"],
            ),
            node(
                func=sample_negatives_dgsr,
                inputs="transactions_mapped",
                outputs="negative_transactions_samples",
                name="sample_negatives_node",
                tags=["preprocess"],
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
            "predict_graphs": f"{namespace}_predict_graphs",
        },
        namespace=namespace,
    )

    return main_pipeline
