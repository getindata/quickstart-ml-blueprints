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
                name="train_val_split_node",
            ),
        ],
        namespace="train_val_split",
        inputs=[
            "train_candidates",
            "val_transactions",],
        outputs=None,
    )