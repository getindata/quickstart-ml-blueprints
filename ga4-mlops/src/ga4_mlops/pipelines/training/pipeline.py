"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    create_calibration_plot,
    evaluate_model,
    fit_calibrator,
    log_metric,
    optimize_xgboost_hyperparameters,
    train_xgboost_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    training_pipeline = pipeline(
        [
            node(
                name="optimize_hyperparameters_node",
                func=optimize_xgboost_hyperparameters,
                inputs=[
                    "train.abt",
                    "valid.abt",
                    "params:seed",
                    "params:optim_time",
                    "params:objective",
                    "params:eval_metric",
                    "params:direction",
                ],
                outputs="best_params",
            ),
            node(
                name="training_node",
                func=train_xgboost_model,
                inputs=[
                    "train.abt",
                    "valid.abt",
                    "best_params",
                ],
                outputs=["fitted.model", "model_config"],
            ),
            node(
                name="fit_calibrator_node",
                func=fit_calibrator,
                inputs=[
                    "test.abt",
                    "fitted.model",
                    "params:calibration_method",
                ],
                outputs="fitted.calibrator",
            ),
            node(
                name="create_calibration_plot_node",
                func=create_calibration_plot,
                inputs=[
                    "test.abt",
                    "fitted.model",
                    "fitted.calibrator",
                ],
                outputs="calibration_plot",
            ),
        ]
    )

    models = ["model", "calibrator"]
    subsets = ["train", "valid", "test"]

    training_and_evaluation_pipeline = training_pipeline

    for model in models:
        for subset in subsets:
            evaluation_pipeline = pipeline(
                [
                    node(
                        name=f"evaluate_model_{model}_{subset}_node",
                        func=evaluate_model,
                        inputs=[
                            f"{subset}.abt",
                            f"fitted.{model}",
                            "params:eval_metric",
                        ],
                        outputs=f"{model}_{subset}_metric_value",
                    ),
                    node(
                        name=f"log_metric_{model}_{subset}_node",
                        func=log_metric,
                        inputs=[
                            f"{model}_{subset}_metric_value",
                            f"params:{subset}_name",
                            f"params:{model}_name",
                        ],
                        outputs=None,
                    ),
                ]
            )
            training_and_evaluation_pipeline += evaluation_pipeline

    return training_and_evaluation_pipeline
