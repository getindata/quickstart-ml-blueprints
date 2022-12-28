"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import concatenate_train_valid_test, get_and_aggregate_data


def create_pipeline(train: bool = True, **kwargs) -> Pipeline:

    train_pipeline = pipeline(
        [
            node(
                name="get_and_aggregate_data_train_node",
                func=get_and_aggregate_data,
                inputs=["ga4_data_train"],
                outputs="df_train",
            ),
        ]
    )

    valid_pipeline = pipeline(
        [
            node(
                name="get_and_aggregate_data_valid_node",
                func=get_and_aggregate_data,
                inputs=["ga4_data_valid"],
                outputs="df_valid",
            ),
        ]
    )

    test_pipeline = pipeline(
        [
            node(
                name="get_and_aggregate_data_test_node",
                func=get_and_aggregate_data,
                inputs=["ga4_data_test"],
                outputs="df_test",
            ),
        ]
    )

    concat_pipeline = pipeline(
        [
            node(
                name="concatenate_train_valid_test_node",
                func=concatenate_train_valid_test,
                inputs=["df_train", "df_valid", "df_test"],
                outputs="df_train_valid_test",
            ),
        ]
    )

    batch_predict_pipeline = pipeline(
        [
            node(
                name="get_and_aggregate_data_batch_predict_node",
                func=get_and_aggregate_data,
                inputs=["ga4_data_batch_predict"],
                outputs="df_batch_predict",
            ),
        ]
    )

    if train:
        return train_pipeline + valid_pipeline + test_pipeline + concat_pipeline
    else:
        return batch_predict_pipeline
