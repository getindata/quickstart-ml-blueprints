from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import filter_out_old_transactions, sample_articles, sample_customers, sample_transactions


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=filter_out_old_transactions,
                inputs=["transactions", "params:sampling.cutoff_date"],
                outputs="latest_transactions",
                name="filter_out_old_transactions_node",
            ),
            node(
                func=sample_articles,
                inputs=["articles", 
                        "params:sampling.article_img_dir",
                        "params:sampling.article_img_sample_dir",
                        "params:sampling.articles_sample_size"],
                outputs="articles_sample",
                name="sample_articles_node",
            ),
            node(
                func=sample_customers,
                inputs=["customers", "latest_transactions", "params:sampling.customers_sample_size"],
                outputs="customers_sample",
                name="sample_customers_node",
            ),
            node(
                func=sample_transactions,
                inputs=["latest_transactions", "customers_sample", "articles_sample"],
                outputs="transactions_sample",
                name="sample_transactions_node",
            ),
        ],
        namespace="data_sampling",
        inputs=["transactions", "articles", "customers"],
        outputs=["transactions_sample", "articles_sample", "customers_sample"],
    )