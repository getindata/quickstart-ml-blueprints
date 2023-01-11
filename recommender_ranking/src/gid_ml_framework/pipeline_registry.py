"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from gid_ml_framework.pipelines import candidate_generation as cg
from gid_ml_framework.pipelines import candidate_generation_validation as cgv
from gid_ml_framework.pipelines import candidates_feature_engineering as cfe
from gid_ml_framework.pipelines import exploratory_data_analysis as eda
from gid_ml_framework.pipelines import feature_engineering_automated as fea
from gid_ml_framework.pipelines import feature_engineering_manual as fem
from gid_ml_framework.pipelines import feature_selection as fs
from gid_ml_framework.pipelines import feature_selection_apply as fsa
from gid_ml_framework.pipelines import filter_latest_transactions as flt
from gid_ml_framework.pipelines import image_embeddings as ie
from gid_ml_framework.pipelines import image_embeddings_inference as iei
from gid_ml_framework.pipelines import image_resizer as ir
from gid_ml_framework.pipelines import merge_candidate_features as mcf
from gid_ml_framework.pipelines import ranking as r
from gid_ml_framework.pipelines import ranking_optuna as ro
from gid_ml_framework.pipelines import recommendation_generation as rg
from gid_ml_framework.pipelines import sample_data as sd
from gid_ml_framework.pipelines import text_embeddings as te
from gid_ml_framework.pipelines import train_val_split as tvs


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    # supplementary pipelines
    sample_data_pipeline = sd.create_pipeline()
    eda_pipeline = eda.create_pipeline()
    image_resizer_pipeline = ir.create_pipeline()

    # image/text embeddings pipelines
    image_embeddings_pipeline = ie.create_pipeline()
    image_embeddings_inference_pipeline = iei.create_pipeline()
    text_embeddings_pipeline = te.create_pipeline()

    # recommendation pipelines
    # training pipelines
    train_filter_latest_transactions_pipeline = flt.create_pipeline(train_flag=True)
    train_candidate_generation_pipeline = cg.create_pipeline(train_flag=True)
    train_feature_engineering_manual_pipeline = fem.create_pipeline(train_flag=True)
    train_feature_engineering_automated_pipeline = fea.create_pipeline(train_flag=True)
    train_candidates_feature_engineering_pipeline = cfe.create_pipeline(train_flag=True)

    # inference pipelines
    inference_filter_latest_transactions_pipeline = flt.create_pipeline(
        train_flag=False
    )
    inference_candidate_generation_pipeline = cg.create_pipeline(train_flag=False)
    inference_feature_engineering_manual_pipeline = fem.create_pipeline(
        train_flag=False
    )
    inference_feature_engineering_automated_pipeline = fea.create_pipeline(
        train_flag=False
    )
    inference_candidates_feature_engineering_pipeline = cfe.create_pipeline(
        train_flag=False
    )

    # training/inference indifferent pipelines
    train_val_split_pipeline = tvs.create_pipeline()
    candidate_generation_validation_pipeline = cgv.create_pipeline()
    feature_selection_pipeline = fs.create_pipeline()
    feature_selection_apply_pipeline = fsa.create_pipeline()
    merge_candidate_features_pipeline = mcf.create_pipeline()
    ranking_pipeline = r.create_pipeline()
    ranking_optuna_pipeline = ro.create_pipeline()
    recommendation_generation_pipeline = rg.create_pipeline()

    return {
        # separate pipelines
        "__default__": sample_data_pipeline,
        "sd": sample_data_pipeline,
        "eda": eda_pipeline,
        "ie": image_embeddings_pipeline,
        "iei": image_embeddings_inference_pipeline,
        "te": text_embeddings_pipeline,
        "ir": image_resizer_pipeline,
        "tvs": train_val_split_pipeline,
        "cgv": candidate_generation_validation_pipeline,
        "fs": feature_selection_pipeline,
        "fsa": feature_selection_apply_pipeline,
        "mcf": merge_candidate_features_pipeline,
        "r": ranking_pipeline,
        "ro": ranking_optuna_pipeline,
        "rg": recommendation_generation_pipeline,
        # connected pipelines
        "feature_engineering_training": (
            train_feature_engineering_manual_pipeline
            + train_feature_engineering_automated_pipeline
            + feature_selection_pipeline
            + feature_selection_apply_pipeline
        ),
        "feature_engineering_inference": (
            inference_feature_engineering_manual_pipeline
            + inference_feature_engineering_automated_pipeline
            + feature_selection_apply_pipeline
        ),
        "train_ranking_model": (merge_candidate_features_pipeline + ranking_pipeline),
        "train_ranking_optuna_model": (
            merge_candidate_features_pipeline + ranking_optuna_pipeline
        ),
        "generate_recommendations": (
            merge_candidate_features_pipeline + recommendation_generation_pipeline
        ),
        # end_to_end models
        # embeddings
        "train_and_generate_embeddings": (
            image_embeddings_pipeline
            + image_embeddings_inference_pipeline
            + text_embeddings_pipeline
        ),
        "generate_embeddings": (
            image_embeddings_inference_pipeline + text_embeddings_pipeline
        ),
        # maximize recall
        "candidate_generation_training": (
            train_filter_latest_transactions_pipeline
            + train_val_split_pipeline
            + train_candidate_generation_pipeline
            + candidate_generation_validation_pipeline
        ),
        # maximize MAP
        "end_to_end_ranking_training": (
            train_filter_latest_transactions_pipeline
            + train_val_split_pipeline
            + train_candidate_generation_pipeline
            + train_feature_engineering_manual_pipeline
            + train_feature_engineering_automated_pipeline
            + feature_selection_pipeline
            + feature_selection_apply_pipeline
            + train_candidates_feature_engineering_pipeline
            + merge_candidate_features_pipeline
            + ranking_pipeline
        ),
        # generate recommendations
        "end_to_end_ranking_inference": (
            inference_filter_latest_transactions_pipeline
            + inference_candidate_generation_pipeline
            + inference_feature_engineering_manual_pipeline
            + inference_feature_engineering_automated_pipeline
            + feature_selection_apply_pipeline
            + inference_candidates_feature_engineering_pipeline
            + merge_candidate_features_pipeline
            + recommendation_generation_pipeline
        ),
    }
