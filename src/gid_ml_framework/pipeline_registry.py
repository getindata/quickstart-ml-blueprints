"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from gid_ml_framework.pipelines import automated_feature_engineering as afe
from gid_ml_framework.pipelines import calculate_image_embeddings as cie
from gid_ml_framework.pipelines import candidate_generation as cg
from gid_ml_framework.pipelines import candidate_generation_validation as cgv
from gid_ml_framework.pipelines import candidates_feature_engineering as cfe
from gid_ml_framework.pipelines import exploratory_data_analysis as eda
from gid_ml_framework.pipelines import graph_recommendation as gr
from gid_ml_framework.pipelines import graph_recommendation_modeling as grm
from gid_ml_framework.pipelines import (
    graph_recommendation_preprocessing as grp,
)
from gid_ml_framework.pipelines import image_embeddings as ie
from gid_ml_framework.pipelines import image_resizer as ir
from gid_ml_framework.pipelines import kaggle_submission as ks
from gid_ml_framework.pipelines import manual_feature_engineering as mfe
from gid_ml_framework.pipelines import merge_candidate_features as mcf
from gid_ml_framework.pipelines import ranking as r
from gid_ml_framework.pipelines import ranking_optuna as ro
from gid_ml_framework.pipelines import recommendation_generation as rg
from gid_ml_framework.pipelines import sample_data as sd
from gid_ml_framework.pipelines import santander_preprocessing as sp
from gid_ml_framework.pipelines import santander_to_act as sta
from gid_ml_framework.pipelines import text_embeddings as te
from gid_ml_framework.pipelines import train_val_split as tvs


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    sample_data_pipeline = sd.create_pipeline()
    eda_pipeline = eda.create_pipeline()
    image_embeddings_pipeline = ie.create_pipeline()
    calculate_image_embeddings_pipeline = cie.create_pipeline()
    text_embeddings_pipeline = te.create_pipeline()
    image_resizer_pipeline = ir.create_pipeline()
    santander_preprocessing_pipeline = sp.create_pipeline()
    santander_to_act_piepline = sta.create_pipeline()
    candidate_generation_pipeline = cg.create_pipeline()
    train_val_split_pipeline = tvs.create_pipeline()
    candidate_generation_validation_pipeline = cgv.create_pipeline()
    manual_feature_engineering_pipeline = mfe.create_pipeline()
    automated_feature_engineering_pipeline = afe.create_pipeline()
    candidates_feature_engineering_pipeline = cfe.create_pipeline()
    merge_candidate_features_pipeline = mcf.create_pipeline()
    ranking_pipeline = r.create_pipeline()
    ranking_optuna_pipeline = ro.create_pipeline()
    recommendation_generation_pipeline = rg.create_pipeline()
    graph_recommendation_preprocessing_santander_pipeline = grp.create_pipeline(
        dataset_namespace="santander"
    )

    graph_recommendation_modeling_santander_dgsr_kaggle_pipeline = grm.create_pipeline(
        dataset="santander",
        model="dgsr",
        comments="kaggle",
    )

    graph_recommendation_santander_dgsr_kaggle_pipeline = gr.create_pipeline(
        dataset="santander",
        model="dgsr",
        comments="kaggle",
    )

    santander_kaggle_submission = ks.create_pipeline(dataset="santander")

    return {
        "__default__": sample_data_pipeline,
        "sd": sample_data_pipeline,
        "eda": eda_pipeline,
        "ie": image_embeddings_pipeline,
        "cie": calculate_image_embeddings_pipeline,
        "te": text_embeddings_pipeline,
        "ir": image_resizer_pipeline,
        "sp": santander_preprocessing_pipeline,
        "sta": santander_to_act_piepline,
        "cg": candidate_generation_pipeline,
        "tvs": train_val_split_pipeline,
        "cgv": candidate_generation_validation_pipeline,
        "candidate_generation": (
            train_val_split_pipeline
            + candidate_generation_pipeline
            + candidate_generation_validation_pipeline
        ),
        "mfe": manual_feature_engineering_pipeline,
        "afe": automated_feature_engineering_pipeline,
        "feature_engineering": (
            manual_feature_engineering_pipeline + automated_feature_engineering_pipeline
        ),
        "cfe": candidates_feature_engineering_pipeline,
        "mcf": merge_candidate_features_pipeline,
        "r": ranking_pipeline,
        "train_ranking_model": (merge_candidate_features_pipeline + ranking_pipeline),
        "ro": ranking_optuna_pipeline,
        "train_ranking_optuna_model": (
            merge_candidate_features_pipeline + ranking_optuna_pipeline
        ),
        "rg": recommendation_generation_pipeline,
        "generate_recommendations": (
            merge_candidate_features_pipeline + recommendation_generation_pipeline
        ),
        "santander_grp": graph_recommendation_preprocessing_santander_pipeline,
        # "hm_grp": graph_recommendation_preprocessing_hm_pipeline,
        # "hm_dgsr_grm": graph_recommendation_modeling_hm_dgsr_pipeline,
        # "hm_dgsr_gr": graph_recommendation_hm_dgsr_pipeline,
        "santander_dgsr_kaggle_grm": graph_recommendation_modeling_santander_dgsr_kaggle_pipeline,
        "santander_dgsr_kaggle_gr": graph_recommendation_santander_dgsr_kaggle_pipeline,
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
    }
