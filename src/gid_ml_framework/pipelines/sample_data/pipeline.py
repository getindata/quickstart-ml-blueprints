from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import filter_out_old_transactions, sample_customers, sample_transactions


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=filter_out_old_transactions,
                inputs=["transactions", "params:cutoff_date"],
                outputs="latest_transactions",
                name="filter_out_old_transactions_node",
            ),
            node(
                func=sample_customers,
                inputs=["customers", "latest_transactions", "params:sample_size"],
                outputs="customers_sample",
                name="sample_customers_node",
            ),
            node(
                func=sample_transactions,
                inputs=["latest_transactions", "customers_sample"],
                outputs="transactions_sample",
                name="sample_transactions_node",
            ),
        ],
        namespace="data_sampling",
        inputs=["transactions", "customers"],
        outputs=["transactions_sample", "customers_sample"],
    )