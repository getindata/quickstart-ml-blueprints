"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from recommender_gnn.pipelines import graph_recommendation as gr
from recommender_gnn.pipelines import graph_recommendation_modeling as grm
from recommender_gnn.pipelines import graph_recommendation_preprocessing as grp
from recommender_gnn.pipelines import kaggle_submission as ks
from recommender_gnn.pipelines import otto_preprocessing as op
from recommender_gnn.pipelines import otto_to_act as ota
from recommender_gnn.pipelines import test_gpu as tg


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    otto_preprocessing_train_pipeline = op.create_pipeline("train")
    otto_preprocessing_test_pipeline = op.create_pipeline("test")
    otto_to_act_train_pipeline = ota.create_pipeline("train")
    otto_to_act_test_pipeline = ota.create_pipeline("test")
    graph_recommendation_preprocessing_hm_pipeline = grp.create_pipeline(
        dataset="hm", transactions_subsets=["train"]
    )
    graph_recommendation_preprocessing_otto_pipeline = grp.create_pipeline(
        dataset="otto", transactions_subsets=["test"]
    )
    graph_recommendation_modeling_hm_dgsr_pipeline = grm.create_pipeline(
        dataset="hm",
        model="dgsr",
    )
    graph_recommendation_modeling_otto_dgsr_pipeline = grm.create_pipeline(
        dataset="otto",
        model="dgsr",
    )
    graph_recommendation_hm_dgsr_pipeline = gr.create_pipeline(
        dataset="hm",
        model="dgsr",
    )
    graph_recommendation_otto_dgsr_pipeline = gr.create_pipeline(
        dataset="otto",
        model="dgsr",
    )
    hm_dgsr_kaggle_submission = ks.create_pipeline(
        dataset="hm",
        model="dgsr",
        users="hm_to_act_train.customers",
    )

    otto_dgsr_kaggle_submission = ks.create_pipeline(
        dataset="otto",
        model="dgsr",
        users="otto_to_act_test.customers",
        test_df="otto_to_act_test.transactions",
    )

    test_gpu_cuda = tg.create_pipeline()

    return {
        "__default__": otto_preprocessing_train_pipeline,
        "otto_preprocessing_train": otto_preprocessing_train_pipeline,
        "otto_preprocessing_test": otto_preprocessing_test_pipeline,
        "otto_to_act_train": otto_to_act_train_pipeline,
        "otto_to_act_test": otto_to_act_test_pipeline,
        "hm_graph_recommendation_preprocessing": graph_recommendation_preprocessing_hm_pipeline,
        "otto_graph_recommendation_preprocessing": graph_recommendation_preprocessing_otto_pipeline,
        "hm_dgsr_graph_recommendation_modeling": graph_recommendation_modeling_hm_dgsr_pipeline,
        "otto_dgsr_graph_recommendation_modeling": graph_recommendation_modeling_otto_dgsr_pipeline,
        "hm_dgsr_graph_recommendation": graph_recommendation_hm_dgsr_pipeline,
        "otto_dgsr_graph_recommendation": graph_recommendation_otto_dgsr_pipeline,
        "hm_dgsr_kaggle_submission": hm_dgsr_kaggle_submission,
        "otto_dgsr_kaggle_submission": otto_dgsr_kaggle_submission,
        "hm_end_to_end": (
            graph_recommendation_preprocessing_hm_pipeline
            + graph_recommendation_modeling_hm_dgsr_pipeline
            + graph_recommendation_hm_dgsr_pipeline
            + hm_dgsr_kaggle_submission
        ),
        "otto_end_to_end": (
            otto_preprocessing_train_pipeline
            + otto_preprocessing_test_pipeline
            + otto_to_act_train_pipeline
            + otto_to_act_test_pipeline
            + graph_recommendation_preprocessing_otto_pipeline
            + graph_recommendation_modeling_otto_dgsr_pipeline
            + graph_recommendation_otto_dgsr_pipeline
            + otto_dgsr_kaggle_submission
        ),
        "tg": test_gpu_cuda,
    }
