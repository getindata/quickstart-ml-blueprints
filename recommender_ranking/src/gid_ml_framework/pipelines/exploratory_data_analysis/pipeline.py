from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import auto_eda, manual_eda


def create_pipeline(**kwargs) -> Pipeline:
    auto_eda_pipeline = pipeline(
        [
            node(
                func=auto_eda,
                inputs=["customers", "params:customers"],
                outputs=None,
                name="customers_node",
            ),
            node(
                func=auto_eda,
                inputs=["articles", "params:articles"],
                outputs=None,
                name="articles_node",
            ),
            node(
                func=auto_eda,
                inputs=["raw_transactions", "params:transactions"],
                outputs=None,
                name="transactions_node",
            ),
        ],
    )

    manual_eda_pipeline = pipeline(
        [
            node(
                func=manual_eda,
                inputs=["articles", "raw_transactions"],
                outputs=None,
                name="manual_eda_node",
            ),
        ],
    )

    return pipeline(
        pipe=auto_eda_pipeline + manual_eda_pipeline,
        inputs=["customers", "articles", "raw_transactions"],
        namespace="exploratory_data_analysis",
    )
