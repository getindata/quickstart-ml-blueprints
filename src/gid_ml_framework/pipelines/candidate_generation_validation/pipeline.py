from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import log_retrieval_recall

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=log_retrieval_recall,
                inputs=[
                    "train_candidates",
                    "val_transactions",
                    ],
                outputs=None,
                name="log_retrieval_recall_node",
            ),
        ],
        namespace="log_retrieval_recall",
        inputs=[
            "train_candidates",
            "val_transactions",],
        outputs=None,
    )