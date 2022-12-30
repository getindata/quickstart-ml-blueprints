"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.4
"""
import logging
import re

import mlflow
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def train_and_validate_model(
    abt_train: pd.DataFrame,
    abt_valid: pd.DataFrame,
    seed: int = 42,
    optim_time: int = 60,
    objective: str = "binary:logistic",
    eval_metric: str = "auc",
    direction: str = "maximize",
):
    """Train and validate XGBoost model, optimize hyperparameters and retrain with best hyperparamer set.

    Args:
        df_train (pd.DataFrame): training data frame
        df_valid (pd.DataFrame): validation data frame
        seed (int, optional): random seed. Defaults to 42.
        optim_time (int, optional): hyperparameter optimization time in seconds. Defaults to 60.
        objective (_type_, optional): objective function type for XGBoost. Defaults to 'binary:logistic'.
        eval_metric (str, optional): model evaluation metric. Defaults to 'auc'.
        direction (str, optional): metric optimization direction. Defaults to 'maximize'.

    Returns:
        XGBoost model
    """
    logger.info(
        "Starting training and validation procedure with hyperparameter optimization..."
    )

    eval_fn = (
        roc_auc_score  # TODO: Can be modified later based on selected `eval_metric`
    )

    num_cols = [item for item in abt_train.columns if re.compile("^n_").match(item)]
    cat_cols = [item for item in abt_train.columns if re.compile("^c_").match(item)]
    target_col = [item for item in abt_train.columns if re.compile("^y_").match(item)][
        0
    ]

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

    # Retrain with best parameters and test model
    all_params = settings
    all_params.update(study.best_params)
    model = xgb.train(
        all_params,
        dtrain,
        evals=[(dvalid, "validation")],
        verbose_eval=False,
        early_stopping_rounds=30,
    )

    preds_train = model.predict(dtrain)
    preds_valid = model.predict(dvalid)
    train_score = eval_fn(abt_train[target_col], preds_train)
    valid_score = eval_fn(abt_valid[target_col], preds_valid)

    hparams = (
        pd.DataFrame.from_dict(all_params, orient="index")
        .reset_index()
        .rename(columns={"index": "hparam", 0: "value"})
    )

    mlflow.log_metric("train_score", train_score)
    mlflow.log_metric("valid_score", valid_score)

    return model, hparams


def test_model(abt_test: pd.DataFrame, model, eval_metric: str = "auc"):
    """Test XGBoost model on the test set.

    Args:
        abt_test (pd.DataFrame): testing data frame
        model (_type_): XGBoost model
        eval_metric (str, optional): model evaluation metric. Defaults to 'auc'.
    """
    logger.info("Testing model performance on the test set...")

    eval_fn = (
        roc_auc_score  # TODO: Can be modified later based on selected `eval_metric`
    )

    num_cols = [item for item in abt_test.columns if re.compile("^n_").match(item)]
    cat_cols = [item for item in abt_test.columns if re.compile("^c_").match(item)]
    target_col = [item for item in abt_test.columns if re.compile("^y_").match(item)][0]

    dtest = xgb.DMatrix(abt_test[num_cols + cat_cols], label=abt_test[[target_col]])

    preds_test = model.predict(dtest)
    test_score = eval_fn(abt_test[target_col], preds_test)

    mlflow.log_metric("test_score", test_score)
