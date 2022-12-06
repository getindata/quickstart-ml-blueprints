"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from recommender_gnn.pipelines import graph_recommendation as gr
from recommender_gnn.pipelines import graph_recommendation_modeling as grm
from recommender_gnn.pipelines import graph_recommendation_preprocessing as grp
from recommender_gnn.pipelines import kaggle_submission as ks
from recommender_gnn.pipelines import otto_preprocessing as op
from recommender_gnn.pipelines import otto_to_act as ota
from recommender_gnn.pipelines import santander_preprocessing as sp
from recommender_gnn.pipelines import santander_to_act as sta
from recommender_gnn.pipelines import test_gpu as tg


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    santander_preprocessing_pipeline = sp.create_pipeline()
    otto_preprocessing_train_pipeline = op.create_pipeline("train")
    otto_preprocessing_test_pipeline = op.create_pipeline("test")
    santander_to_act_piepline = sta.create_pipeline()
    otto_to_act_train_pipeline = ota.create_pipeline("train")
    otto_to_act_test_pipeline = ota.create_pipeline("test")
    graph_recommendation_preprocessing_santander_pipeline = grp.create_pipeline(
        dataset="santander",
        train_subset=True,
        val_subset=True,
    )
    graph_recommendation_preprocessing_otto_pipeline = grp.create_pipeline(
        dataset="otto",
        train_subset=False,
        val_subset=True,
    )
    graph_recommendation_modeling_santander_dgsr_kaggle_pipeline = grm.create_pipeline(
        dataset="santander",
        model="dgsr",
        comments="kaggle",
    )
    graph_recommendation_modeling_otto_dgsr_pipeline = grm.create_pipeline(
        dataset="otto",
        model="dgsr",
    )
    graph_recommendation_santander_dgsr_kaggle_pipeline = gr.create_pipeline(
        dataset="santander",
        model="dgsr",
        comments="kaggle",
    )
    graph_recommendation_otto_dgsr_pipeline = gr.create_pipeline(
        dataset="otto",
        model="dgsr",
    )
    santander_kaggle_submission = ks.create_pipeline(dataset="santander")

    test_gpu_cuda = tg.create_pipeline()

    return {
        "__default__": santander_preprocessing_pipeline,
        "sp": santander_preprocessing_pipeline,
        "op_train": otto_preprocessing_train_pipeline,
        "op_test": otto_preprocessing_test_pipeline,
        "sta": santander_to_act_piepline,
        "ota_train": otto_to_act_train_pipeline,
        "ota_test": otto_to_act_test_pipeline,
        "santander_grp": graph_recommendation_preprocessing_santander_pipeline,
        "otto_grp": graph_recommendation_preprocessing_otto_pipeline,
        # "hm_grp": graph_recommendation_preprocessing_hm_pipeline,
        # "hm_dgsr_grm": graph_recommendation_modeling_hm_dgsr_pipeline,
        # "hm_dgsr_gr": graph_recommendation_hm_dgsr_pipeline,
        "santander_dgsr_kaggle_grm": graph_recommendation_modeling_santander_dgsr_kaggle_pipeline,
        "otto_dgsr_grm": graph_recommendation_modeling_otto_dgsr_pipeline,
        "santander_dgsr_kaggle_gr": graph_recommendation_santander_dgsr_kaggle_pipeline,
        "otto_dgsr_gr": graph_recommendation_otto_dgsr_pipeline,
        "santander_ks": santander_kaggle_submission,
        # "hm_dgsr_kaggle_grm": graph_recommendation_modeling_hm_dgsr_kaggle_pipeline,
        # "hm_dgsr_kaggle_gr": graph_recommendation_hm_dgsr_kaggle_pipeline,
        # "hm_ks": hm_kaggle_submission,
        "santander_e2e": (
            santander_preprocessing_pipeline
            + santander_to_act_piepline
            + graph_recommendation_preprocessing_santander_pipeline
            + graph_recommendation_modeling_santander_dgsr_kaggle_pipeline
            + graph_recommendation_santander_dgsr_kaggle_pipeline
            + santander_kaggle_submission
        ),
        "tg": test_gpu_cuda,
    }
