from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    filter_out_old_transactions,
    sample_articles,
    sample_customers,
    sample_transactions,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=filter_out_old_transactions,
                inputs=["hm_transactions_input", "params:cutoff_date"],
                outputs="latest_transactions",
                name="filter_out_old_transactions_node",
            ),
            node(
                func=sample_articles,
                inputs=[
                    "hm_articles_input",
                    "params:article_img_dir",
                    "params:article_img_sample_dir",
                    "params:articles_sample_size",
                ],
                outputs="hm_articles",
                name="sample_articles_node",
            ),
            node(
                func=sample_customers,
                inputs=[
                    "hm_customers_input",
                    "latest_transactions",
                    "params:customers_sample_size",
                ],
                outputs="hm_customers",
                name="sample_customers_node",
            ),
            node(
                func=sample_transactions,
                inputs=["latest_transactions", "hm_customers", "hm_articles"],
                outputs="hm_transactions",
                name="sample_transactions_node",
            ),
        ],
        namespace="data_sampling",
        inputs=["hm_transactions_input", "hm_articles_input", "hm_customers_input"],
        outputs=["hm_transactions", "hm_articles", "hm_customers"],
    )
