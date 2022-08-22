from .nodes import (impute_santander, sample_santander, filter_santander, clean_santander,
                    split_santander,)
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance_train = pipeline(
        [
            node(
                func=sample_santander,
                inputs=["santander_train_input",
                        "params:santander_preprocessing_train.sample.sample_user_frac",
                        "params:santander_preprocessing_train.sample.cutoff_date"],
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
                inputs=["santander_pre_train",
                        "params:santander_preprocessing_train.impute.test"],
                outputs="santander_train",
                name="impute_train_santander_node",
            ),
            node(
                func=impute_santander,
                inputs=["santander_pre_val",
                        "params:santander_preprocessing_train.impute.test"],
                outputs="santander_val",
                name="impute_val_santander_node",
            ),
        ]
    )

    pipeline_instance_test = pipeline(
        [
            node(
                func=sample_santander,
                inputs=["santander_test_input",
                        "params:santander_preprocessing_test.sample.sample_user_frac",
                        "params:santander_preprocessing_test.sample.cutoff_date"],
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
                inputs=["santander_cleaned",
                        "params:santander_preprocessing_test.impute.test"],
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