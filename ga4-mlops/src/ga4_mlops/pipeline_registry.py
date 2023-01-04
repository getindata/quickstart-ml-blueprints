"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines import (
    data_preprocessing,
    feature_engineering,
    prediction,
    training,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_preprocessing_train_pipeline = data_preprocessing.create_pipeline(
        subset="train"
    )
    data_preprocessing_valid_pipeline = data_preprocessing.create_pipeline(
        subset="valid"
    )
    data_preprocessing_test_pipeline = data_preprocessing.create_pipeline(subset="test")
    data_preprocessing_predict_pipeline = data_preprocessing.create_pipeline(
        subset="predict"
    )

    feature_engineering_train_pipeline = feature_engineering.create_pipeline(
        subset="train"
    )
    feature_engineering_valid_pipeline = feature_engineering.create_pipeline(
        subset="valid"
    )
    feature_engineering_test_pipeline = feature_engineering.create_pipeline(
        subset="test"
    )
    feature_engineering_predict_pipeline = feature_engineering.create_pipeline(
        subset="predict"
    )

    training_pipeline = training.create_pipeline()
    prediction_pipeline = prediction.create_pipeline()

    data_preprocessing_train_valid_test_pipeline = (
        data_preprocessing_train_pipeline
        + data_preprocessing_valid_pipeline
        + data_preprocessing_test_pipeline
    )

    feature_engineering_train_valid_test_pipeline = (
        feature_engineering_train_pipeline
        + feature_engineering_valid_pipeline
        + feature_engineering_test_pipeline
    )

    end_to_end_training_pipeline = (
        data_preprocessing_train_valid_test_pipeline
        + feature_engineering_train_valid_test_pipeline
        + training_pipeline
    )

    end_to_end_prediction_pipeline = (
        data_preprocessing_predict_pipeline
        + feature_engineering_predict_pipeline
        + prediction_pipeline
    )

    return {
        "__default__": data_preprocessing_train_pipeline,
        "data_preprocessing_train": data_preprocessing_train_pipeline,
        "data_preprocessing_valid": data_preprocessing_valid_pipeline,
        "data_preprocessing_test": data_preprocessing_test_pipeline,
        "data_preprocessing_train_valid_test": data_preprocessing_train_valid_test_pipeline,
        "data_preprocessing_predict": data_preprocessing_predict_pipeline,
        "feature_engineering_train": feature_engineering_train_pipeline,
        "feature_engineering_valid": feature_engineering_valid_pipeline,
        "feature_engineering_test": feature_engineering_test_pipeline,
        "feature_engineering_train_valid_test": feature_engineering_train_valid_test_pipeline,
        "feature_engineering_predict": feature_engineering_predict_pipeline,
        "training": training_pipeline,
        "prediction": prediction_pipeline,
        "end_to_end_training": end_to_end_training_pipeline,
        "end_to_end_prediction": end_to_end_prediction_pipeline,
    }
