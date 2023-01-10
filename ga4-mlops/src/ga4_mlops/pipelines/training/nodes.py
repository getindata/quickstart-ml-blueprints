"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.4
"""
import json
import logging
from typing import Tuple

import mlflow
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from ..data_preparation_utils import extract_column_names

logger = logging.getLogger(__name__)


def optimize_hyperparameters(
    abt_train: pd.DataFrame,
    abt_valid: pd.DataFrame,
    seed: int = 42,
    optim_time: int = 60,
    objective: str = "binary:logistic",
    eval_metric: str = "auc",
    direction: str = "maximize",
) -> dict:
    """Optimize hyperparameters for XGBoost model.

    Args:
        df_train (pd.DataFrame): training data frame
        df_valid (pd.DataFrame): validation data frame
        seed (int, optional): random seed. Defaults to 42.
        optim_time (int, optional): hyperparameter optimization time in seconds. Defaults to 60.
        objective (_type_, optional): objective function type for XGBoost. Defaults to 'binary:logistic'.
        eval_metric (str, optional): model evaluation metric. Defaults to 'auc'.
        direction (str, optional): metric optimization direction. Defaults to 'maximize'.

    Returns:
        dict: dictionary with settings and best hyperparameters from XGBoost optimization
    """
    logger.info(
        "Starting training and validation procedure with hyperparameter optimization..."
    )

    eval_fn = _get_eval_fn(eval_metric)

    _, num_cols, cat_cols, target_col = extract_column_names(abt_train)
    dtrain = xgb.DMatrix(abt_train[num_cols + cat_cols], label=abt_train[[target_col]])
    dvalid = xgb.DMatrix(abt_valid[num_cols + cat_cols], label=abt_valid[[target_col]])

    settings = {
        "num_boost_round": 200,
        "seed": seed,
        "verbosity": 0,
        "objective": objective,
        "eval_metric": eval_metric,
    }

    def xgb_objective(trial):
        params = {
            "booster": trial.suggest_categorical("booster", ["gbtree"]),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "eta": trial.suggest_float("eta", 1e-6, 1.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
        }

        all_params = settings
        all_params.update(params)

        model = xgb.train(
            all_params,
            dtrain,
            evals=[(dvalid, "validation")],
            verbose_eval=False,
            early_stopping_rounds=30,
        )

        valid_preds = model.predict(dvalid)
        valid_score = eval_fn(abt_valid[target_col], valid_preds)

        return valid_score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)

    study.optimize(xgb_objective, timeout=optim_time)

    best_params = settings
    best_params.update(study.best_params)

    return best_params


def train_and_validate_model(
    abt_train: pd.DataFrame,
    abt_valid: pd.DataFrame,
    hparams: dict,
    eval_metric: str = "auc",
) -> Tuple[xgb.Booster, str]:
    """Trains and validates XGBoost model.

    Args:
        abt_train (pd.DataFrame): training data frame
        abt_valid (pd.DataFrame): validation data frame
        hparams (dict): XGBoost settings and hyperparameters as returned by `optimize_hyperparameters`
        eval_metric (str, optional): model evaluation metric. Defaults to "auc".

    Returns:
        Tuple[xgb.Booster, str]: trained XGBoost model and string with XGBoost full config in JSON-like format
    """
    logger.info("Training and validating XGBoost model...")

    eval_fn = _get_eval_fn(eval_metric)

    _, num_cols, cat_cols, target_col = extract_column_names(abt_train)
    dtrain = xgb.DMatrix(abt_train[num_cols + cat_cols], label=abt_train[[target_col]])
    dvalid = xgb.DMatrix(abt_valid[num_cols + cat_cols], label=abt_valid[[target_col]])

    model = xgb.train(
        hparams,
        dtrain,
        evals=[(dvalid, "validation")],
        verbose_eval=False,
        early_stopping_rounds=30,
    )

    preds_train = model.predict(dtrain)
    preds_valid = model.predict(dvalid)
    train_score = eval_fn(abt_train[target_col], preds_train)
    valid_score = eval_fn(abt_valid[target_col], preds_valid)

    model_config = json.loads(model.save_config())

    mlflow.log_metric("train_score", train_score)
    mlflow.log_metric("valid_score", valid_score)

    return model, model_config


def test_model(abt_test: pd.DataFrame, model, eval_metric: str = "auc"):
    """Test XGBoost model on the test set.

    Args:
        abt_test (pd.DataFrame): testing data frame
        model: XGBoost model
        eval_metric (str, optional): model evaluation metric. Defaults to 'auc'.
    """
    logger.info("Testing model performance on the test set...")

    eval_fn = _get_eval_fn(eval_metric)
    _, num_cols, cat_cols, target_col = extract_column_names(abt_test)
    dtest = xgb.DMatrix(abt_test[num_cols + cat_cols], label=abt_test[[target_col]])

    preds_test = model.predict(dtest)
    test_score = eval_fn(abt_test[target_col], preds_test)

    mlflow.log_metric("test_score", test_score)


def _get_eval_fn(eval_metric: str):
    """Get evaluation function based on metric name.

    Args:
        eval_metric (str): evaluation metric name

    Returns:
        Evaluation function.
    """
    # TODO: Add different metrics
    # If label based, need to add threshold based labeling for predicted scores
    allowed_metrics = ["auc"]
    assert (
        eval_metric in allowed_metrics
    ), f"Evaluation metric has to one of: {allowed_metrics}"

    if eval_metric == "auc":
        eval_fn = roc_auc_score
    else:
        eval_fn = None

    return eval_fn
