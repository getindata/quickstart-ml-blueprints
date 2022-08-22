from .nodes import (impute_santander, sample_santander, filter_santander, clean_santander,
                    split_santander,)
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance = pipeline(
        [
            node(
                func=sample_santander,
                inputs=["santander_train",
                        "params:sample.sample_user_frac",
                        "params:sample.cutoff_date"],
                outputs="santander_sample",
                name="sample_santander_node",
                tags=["train", "test"],
            ),
            node(
                func=filter_santander,
                inputs="santander_sample",
                outputs="santander_filtered",
                name="filter_santander_node",
                tags=["train", "test"],
            ),
            node(
                func=clean_santander,
                inputs="santander_filtered",
                outputs="santander_cleaned",
                name="clean_santander_node",
                tags=["train", "test"],
            ),
            node(
                func=split_santander,
                inputs="santander_cleaned",
                outputs=["santander_pre_train", "santander_pre_val"],
                name="split_santander_node",
                tags=["train"],
            ),
            node(
                func=impute_santander,
                inputs=["santander_pre_train",
                        "params:impute.test"],
                outputs="santander_train",
                name="impute_train_santander_node",
                tags=["train"],
            ),
            node(
                func=impute_santander,
                inputs="santander_pre_val",
                outputs="santander_val",
                name="impute_val_santander_node",
                tags=["train"],
            ),
            node(
                func=impute_santander,
                inputs="santander_cleaned",
                outputs="santander_test",
                name="impute_test_santander_node",
                tags=["test"],
            ),
        ]
    )

    train_nodes = pipeline_instance.only_nodes_with_tags('train')
    test_nodes = pipeline_instance.only_nodes_with_tags('test')

    train_pipeline = pipeline(
        pipe=train_nodes,
        inputs="santander_train",
        outputs=["santander_train", "santander_val"],
        namespace="santander_preprocessing_train",
    )

    return train_pipeline