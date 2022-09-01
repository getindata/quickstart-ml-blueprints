from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    clean_santander,
    filter_santander,
    impute_santander,
    sample_santander,
    split_santander,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance_train = pipeline(
        [
            node(
                func=sample_santander,
                inputs=[
                    "santander_train_input",
                    "params:sample.sample_user_frac",
                    "params:sample.cutoff_date",
                    "params:sample.stratify",
                ],
                outputs="santander_sample",
                name="sample_santander_node",
            ),
            node(
                func=filter_santander,
                inputs="santander_sample",
                outputs="santander_filtered",
                name="filter_santander_node",
            ),
            node(
                func=clean_santander,
                inputs="santander_filtered",
                outputs="santander_cleaned",
                name="clean_santander_node",
            ),
            node(
                func=split_santander,
                inputs="santander_cleaned",
                outputs=["santander_pre_train", "santander_pre_val"],
                name="split_santander_node",
            ),
            node(
                func=impute_santander,
                inputs=["santander_pre_train", f"params:impute.test"],
                outputs="santander_train",
                name="impute_train_santander_node",
            ),
            node(
                func=impute_santander,
                inputs=["santander_pre_val", f"params:impute.test"],
                outputs="santander_val",
                name="impute_val_santander_node",
            ),
        ]
    )

    pipeline_instance_test = pipeline(
        [
            node(
                func=sample_santander,
                inputs=[
                    "santander_test_input",
                    "params:sample.sample_user_frac",
                    "params:sample.cutoff_date",
                    "params:sample.stratify",
                ],
                outputs="santander_sample",
                name="sample_santander_node",
            ),
            node(
                func=filter_santander,
                inputs="santander_sample",
                outputs="santander_filtered",
                name="filter_santander_node",
            ),
            node(
                func=clean_santander,
                inputs="santander_filtered",
                outputs="santander_cleaned",
                name="clean_santander_node",
            ),
            node(
                func=impute_santander,
                inputs=["santander_cleaned", "params:impute.test"],
                outputs="santander_test",
                name="impute_test_santander_node",
            ),
        ]
    )

    train_pipeline = pipeline(
        pipe=pipeline_instance_train,
        inputs=["santander_train_input"],
        outputs=["santander_train", "santander_val"],
        namespace="santander_preprocessing_train",
    )

    test_pipeline = pipeline(
        pipe=pipeline_instance_test,
        inputs=["santander_test_input"],
        outputs="santander_test",
        namespace="santander_preprocessing_test",
    )

    return train_pipeline + test_pipeline
