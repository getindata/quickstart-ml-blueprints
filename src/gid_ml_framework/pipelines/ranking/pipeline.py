from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import train_val_split, train_single_model


def create_pipeline(**kwargs) -> Pipeline:
    train_single_ranking_model_pipeline = pipeline(
        [
            node(
                func=train_val_split,
                name="train_val_split_node",
                inputs=[
                    "final_candidates",
                    "params:dataset.val_size",
                    "params:dataset.downsampling",
                    "params:dataset.neg_samples",
                    ],
                outputs=["train_candidates", "val_candidates"],
            ),
            node(
                func=train_single_model,
                name="train_single_model_node",
                inputs=[
                    "train_candidates",
                    "val_candidates",
                    "val_transactions",
                    "params:training.params",
                    "params:training.k",
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
    
    return train_single_ranking_model_pipeline
