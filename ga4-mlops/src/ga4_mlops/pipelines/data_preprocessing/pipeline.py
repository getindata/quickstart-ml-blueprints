"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import get_and_aggregate_data


def create_pipeline(train: bool = True, **kwargs) -> Pipeline:

    train_pipeline = pipeline(
        [
            node(
                name="get_and_aggregate_data_train_node",
                func=get_and_aggregate_data,
                inputs=["ga4_data_train"],
                outputs=None,
            ),
        ]
    )

    valid_pipeline = pipeline(
        [
            node(
                name="get_and_aggregate_data_valid_node",
                func=get_and_aggregate_data,
                inputs=["ga4_data_valid"],
                outputs=None,
            ),
        ]
    )

    test_pipeline = pipeline(
        [
            node(
                name="get_and_aggregate_data_test_node",
                func=get_and_aggregate_data,
                inputs=["ga4_data_test"],
                outputs=None,
            ),
        ]
    )

    batch_predict_pipeline = pipeline(
        [
            node(
                name="get_and_aggregate_data_batch_predict_node",
                func=get_and_aggregate_data,
                inputs=["ga4_data_batch_predict"],
                outputs=None,
            ),
        ]
    )

    if train:
        return train_pipeline + valid_pipeline + test_pipeline
    else:
        return batch_predict_pipeline
