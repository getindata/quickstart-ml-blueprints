"""
This is a boilerplate pipeline 'santander_preprocessing'
generated using Kedro 0.17.7
"""

from gid_ml_framework.pipelines.santander_preprocessing.nodes import sample_santander
from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=sample_santander,
                inputs=["santander",
                        "params:santander_preprocessing.sample_user_frac",
                        "params:santander_preprocessing.cutoff_date"],
                outputs="santander_sample",
                name="sample_santander_node",
            ),
        ],
        namespace="santander_preprocessing",
        inputs=["santander"],
        outputs=["santander_sample"],
    )
