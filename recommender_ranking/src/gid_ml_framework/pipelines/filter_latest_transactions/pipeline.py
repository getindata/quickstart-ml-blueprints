from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import filter_dataframe_by_last_n_weeks


def create_pipeline(train_flag: bool = False, **kwargs) -> Pipeline:
    "Use string 'inference' if inference, or 'train' if training flag"
    mode_prefix = "train" if train_flag else "inference"

    return pipeline(
        [
            node(
                func=filter_dataframe_by_last_n_weeks,
                inputs=[
                    "raw_transactions",
                    "params:date_column",
                    f"params:{mode_prefix}.no_weeks",
                ],
                outputs="transactions",
                name="filter_dataframe_by_last_n_weeks_node",
            ),
        ],
        namespace="filter_latest_transactions",
        inputs=["raw_transactions"],
        outputs=["transactions"],
    )
