from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import train_val_split, train_model


def create_pipeline(**kwargs) -> Pipeline:
    train_ranking_model_pipeline = pipeline(
        [
            node(
                func=train_val_split,
                name="train_val_split_node",
                inputs=[
                    "final_candidates",
                    "params:ranking.val_size",
                    ],
                outputs=["train_candidates", "val_candidates"],
            ),
            node(
                func=train_model,
                name="train_model_node",
                inputs=[
                    "train_candidates",
                    "val_candidates",
                    "params:ranking.training.params",
                    "val_transactions",
                    "params:ranking.training.k"
                    ],
                outputs=None,
            ),
        ],
        namespace="train_ranking",
        inputs=[
            "final_candidates",
            "val_transactions",
        ],
        outputs=None,
    )
    
    return train_ranking_model_pipeline
