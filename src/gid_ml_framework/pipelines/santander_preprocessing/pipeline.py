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
    train_namespace_params = "params:santander_preprocessing_train"
    test_namespace_params = "params:santander_preprocessing_test"

    pipeline_instance_train = pipeline(
        [
            node(
                func=sample_santander,
                inputs=[
                    "santander_train_input",
                    f"{train_namespace_params}.sample.sample_user_frac",
                    f"{train_namespace_params}.sample.cutoff_date",
                    f"{train_namespace_params}.sample.stratify",
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
                inputs=["santander_pre_train", f"{train_namespace_params}.impute.test"],
                outputs="santander_train",
                name="impute_train_santander_node",
            ),
            node(
                func=impute_santander,
                inputs=["santander_pre_val", f"{train_namespace_params}.impute.test"],
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
                    f"{test_namespace_params}.sample.sample_user_frac",
                    f"{test_namespace_params}.sample.cutoff_date",
                    f"{test_namespace_params}.sample.stratify",
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
                inputs=["santander_cleaned", f"{test_namespace_params}.impute.test"],
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
