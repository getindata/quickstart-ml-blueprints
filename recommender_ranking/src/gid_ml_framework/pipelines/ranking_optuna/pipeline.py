from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from gid_ml_framework.pipelines.ranking.nodes import (
    train_optuna_model,
    train_val_split,
)


def create_pipeline(**kwargs) -> Pipeline:
    train_optuna_ranking_model_pipeline = pipeline(
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
                func=train_optuna_model,
                name="train_optuna_model_node",
                inputs=[
                    "train_candidates",
                    "val_candidates",
                    "val_transactions",
                    "params:training.params",
                    "params:training.optuna_params",
                    "params:training.k",
                ],
                outputs=None,
            ),
        ],
        namespace="train_optuna_ranking",
        inputs=[
            "final_candidates",
            "val_transactions",
        ],
        outputs=None,
    )

    return train_optuna_ranking_model_pipeline
