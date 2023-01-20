"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.4
"""
import logging
from typing import Any, Callable, Tuple

import mlflow
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from ..data_preparation_utils import extract_column_names

logger = logging.getLogger(__name__)


def optimize_xgboost_hyperparameters(
    abt_train: pd.DataFrame,
    abt_valid: pd.DataFrame,
    seed: int = 42,
    optim_time: int = 60,
    objective: str = "binary:logistic",
    eval_metric: str = "auc",
    direction: str = "maximize",
) -> dict:
    """Optimize hyperparameters for XGBoost model.
    Optimization procedure uses native XGBoost API with DMatrix datasets to speed up the procedure.

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
        "n_estimators": 200,
        "seed": seed,
        "verbosity": 0,
        "objective": objective,
        "eval_metric": eval_metric,
        "early_stopping_rounds": 30,
    }

    def objective(trial):
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

    study.optimize(objective, timeout=optim_time)

    best_params = settings
    best_params.update(study.best_params)

    return best_params


def train_xgboost_model(
    abt_train: pd.DataFrame,
    abt_valid: pd.DataFrame,
    hparams: dict,
) -> Tuple[XGBClassifier, str]:
    """Trains and validates XGBoost model.
    Final training procedure uses scikit-learn API to be compatible with XAI packages
    and consistent with other scikit-learn algorithms.

    Args:
        abt_train (pd.DataFrame): training data frame
        abt_valid (pd.DataFrame): validation data frame
        hparams (dict): XGBoost settings and hyperparameters as returned by `optimize_hyperparameters`

    Returns:
        Tuple[XGBClassifier, str]: trained XGBoost model and string with XGBoost full config in JSON-like format
    """
    logger.info("Training and validating XGBoost model...")

    _, num_cols, cat_cols, target_col = extract_column_names(abt_train)

    model = XGBClassifier(**hparams)
    model.fit(
        X=abt_train[num_cols + cat_cols],
        y=abt_train[target_col],
        eval_set=[(abt_valid[num_cols + cat_cols], abt_valid[target_col])],
        verbose=False,
    )

    model_config = model.get_params()

    return model, model_config


def fit_calibrator(
    abt_test: pd.DataFrame, model: Any, method: str = "isotonic"
) -> CalibratedClassifierCV:
    """_summary_

    Args:
        abt_test (pd.DataFrame): _description_
        model (Any): _description_
        method (str, optional): _description_. Defaults to "isotonic".

    Returns:
        CalibratedClassifierCV: _description_
    """
    calibration_methods = ["isotonic", "sigmoid"]
    assert (
        method in calibration_methods
    ), f"Parameter `method` should be one of: {calibration_methods}"

    _, num_cols, cat_cols, target_col = extract_column_names(abt_test)

    calibrator = CalibratedClassifierCV(model, method=method, cv="prefit")

    calibrator.fit(X=abt_test[num_cols + cat_cols], y=abt_test[target_col])

    return calibrator


def evaluate_model(abt: pd.DataFrame, model: Any, eval_metric: str = "auc") -> float:
    """Test XGBoost model on the test set.

    Args:
        abt_test (pd.DataFrame): ABT to evaluate on
        model (Any): any model with `predict_proba` method
        eval_metric (str, optional): model evaluation metric. Defaults to 'auc'.

    Returns:

    """
    logger.info("Testing model performance on the test set...")

    eval_fn = _get_eval_fn(eval_metric)
    _, num_cols, cat_cols, target_col = extract_column_names(abt)

    preds = model.predict_proba(abt[num_cols + cat_cols])[:, 1]
    metric_value = eval_fn(abt[target_col], preds)

    return metric_value


def log_metric(metric_value: float, subset_alias: str, model_alias: str) -> None:
    """Log metric value for given subset to MLflow.

    Args:
        metric_value (float): metric value
        subset_alias (str): subset alias ("train", "valid", "test")
        model_alias (str): model alias ("model", "calibrator")
    """
    logger.info(f"Logging {subset_alias} metric value for {model_alias}...")

    mlflow.log_metric(f"{model_alias}_{subset_alias}_metric_value", metric_value)


def _get_eval_fn(eval_metric: str) -> Callable:
    """Get evaluation function based on metric name.

    Args:
        eval_metric (str): evaluation metric name

    Returns:
        Callable: evaluation function
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
