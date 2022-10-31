import logging
from typing import Dict, List, Tuple

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import optuna
import pandas as pd
from optuna.integration.mlflow import MLflowCallback
from sklearn.model_selection import train_test_split

from gid_ml_framework.helpers.metrics import map_at_k

logger = logging.getLogger(__name__)


def _prepare_groups_for_ranking(candidates: pd.DataFrame) -> pd.DataFrame:
    """Prepares 'group' for LightGBM dataset. For learning to rank tasks, it is required.
    https://lightgbm.readthedocs.io/en/latest/Parameters.html#query-data

    Args:
        candidates (pd.DataFrame): candidates

    Returns:
        pd.DataFrame: groups
    """
    candidates_group = candidates[["customer_id", "article_id"]]
    candidates_group = candidates_group.groupby(["customer_id"]).size().values
    return candidates_group


def _prepare_lgb_dataset(
    candidates: pd.DataFrame,
    label: str,
    features: List[str],
    cat_features: List[str],
    group: pd.DataFrame = None,
) -> lgb.Dataset:
    """Prepares LightGBM dataset for training a model depending on whether it's
    a learning to rank (group) or other task.

    Args:
        candidates (pd.DataFrame): candidates
        label (str): label name
        features (List[str]): all features
        cat_features (List[str]): categorical features
        group (pd.DataFrame, optional): group dataframe. Defaults to None.

    Returns:
        lgb.Dataset: dataset for training a model
    """
    if group is not None:
        lgb_dataset = lgb.Dataset(
            data=candidates[features],
            label=candidates[label],
            group=group,
            feature_name=features,
            categorical_feature=cat_features,
            free_raw_data=False,
        )
    else:
        lgb_dataset = lgb.Dataset(
            data=candidates[features],
            label=candidates[label],
            feature_name=features,
            categorical_feature=cat_features,
            free_raw_data=False,
        )
    return lgb_dataset


def train_val_split(
    candidates: pd.DataFrame,
    val_size: float = 0.15,
    downsampling: bool = True,
    neg_sample: int = 1_000_000,
) -> Tuple[pd.DataFrame]:
    """Splits dataset into training and validation set. It uses stratification based on whether bought any item or not.

    Args:
        candidates (pd.DataFrame): candidates
        val_size (float, optional): validation set size. Defaults to 0.15.

    Returns:
        Tuple[pd.DataFrame]: tuple of training and validation datasets respectively
    """
    df_split = candidates.groupby(["customer_id"])["label"].max().reset_index()
    train_candidates, val_candidates = train_test_split(
        df_split, test_size=val_size, random_state=42, stratify=df_split["label"]
    )
    train_candidates = candidates[
        candidates["customer_id"].isin(train_candidates["customer_id"].unique())
    ]
    val_candidates = candidates[
        candidates["customer_id"].isin(val_candidates["customer_id"].unique())
    ]
    logger.info(
        f"Train candidates shape: {train_candidates.shape}, \nval candidates shape{val_candidates.shape}"
    )
    if downsampling:
        # downsampling
        logger.info(
            f"Downsampling candidates. Number of added negative samples to the training set: {neg_sample}"
        )
        train_candidates = pd.concat(
            [
                train_candidates[train_candidates.label > 0],  # positive samples
                train_candidates[train_candidates.label == 0].sample(
                    n=neg_sample, random_state=42
                ),
            ],  # negative samples
            axis=0,
        )
        logger.info(train_candidates.label.value_counts(normalize=False))
    return train_candidates, val_candidates


def _predict(model: lgb.Booster, candidates: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    """Predicts the label given the LightGBM model and candidates dataframe.

    Args:
        model (lgb.Booster): LightGBM model
        candidates (pd.DataFrame): candidates
        k (int, optional): top k recommended items. Defaults to 12.

    Returns:
        pd.DataFrame: dataframe with customer_id and list of `k` items most likely to be bought by the customer.
    """
    candidates_temp = candidates.copy()
    candidates_temp["prob"] = model.predict(
        candidates_temp.drop(["customer_id", "article_id", "label"], axis=1)
    )
    pred_lgb = (
        candidates_temp[["customer_id", "article_id", "prob"]]
        .sort_values(by=["customer_id", "prob"], ascending=False)
        .reset_index(drop=True)
    )
    pred_lgb = pred_lgb.groupby(["customer_id"]).head(k)
    return pred_lgb.groupby(["customer_id"])["article_id"].apply(list).reset_index()


def _calculate_map(
    predictions: pd.DataFrame, val_transactions: pd.DataFrame, k: int = 12
) -> float:
    """Calculates mean average precision at `k`.

    Args:
        predictions (pd.DataFrame): dataframe with customer_id and list of recommended items
        val_transactions (pd.DataFrame): validation transactions
        k (int, optional): top k recommended items. Defaults to 12.

    Returns:
        float: mean average precision at `k`
    """
    df_map = (
        val_transactions.groupby(["customer_id"])["article_id"]
        .apply(list)
        .reset_index()
        .merge(predictions, on="customer_id", how="inner")
    )
    df_map.columns = ["customer_id", "y_true", "y_pred"]
    return map_at_k(df_map["y_true"], df_map["y_pred"], k=k)


def train_single_model(
    train_candidates: pd.DataFrame,
    val_candidates: pd.DataFrame,
    val_transactions: pd.DataFrame,
    model_params: Dict,
    k: int = 12,
) -> None:
    """Trains a LightGBM model, (optionally) logs the model and metrics to MLflow.

    Args:
        train_candidates (pd.DataFrame): training candidates
        val_candidates (pd.DataFrame): validation candidates
        val_transactions (pd.DataFrame): validation transactions
        model_params (Dict): parameters for LightGBM model
        k (int, optional): top k recommended items. Defaults to 12.
    """
    logger.info(f"Train positive rate: {train_candidates.label.mean()}")
    train_data_size = len(train_candidates)
    features = [
        col
        for col in train_candidates.columns
        if col not in ["label", "customer_id", "article_id"]
    ]
    cat_features = train_candidates.select_dtypes(include="category").columns.to_list()
    logger.info(f"Categorical features: {cat_features}")
    # train dataset (if ranking then groups)
    train_group = (
        _prepare_groups_for_ranking(train_candidates)
        if model_params["objective"] == "lambdarank"
        else None
    )
    train_dataset = _prepare_lgb_dataset(
        train_candidates, "label", features, cat_features, train_group
    )
    # val dataset (if ranking then groups)
    val_group = (
        _prepare_groups_for_ranking(val_candidates)
        if model_params["objective"] == "lambdarank"
        else None
    )
    val_dataset = _prepare_lgb_dataset(
        val_candidates, "label", features, cat_features, val_group
    )

    mlflow.lightgbm.autolog(silent=True)
    logger.info(f'Starting training model for objective: {model_params["objective"]}')
    model = lgb.train(
        model_params,
        train_dataset,
        valid_sets=[train_dataset, val_dataset],
        valid_names=["train", "valid"],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=20)],
    )
    # train_data_size
    mlflow.log_metric("training_dataset_size", train_data_size)
    # train loss
    logger.info("Recommending for training candidates")
    train_predictions = _predict(model, train_candidates, k)
    train_map = _calculate_map(train_predictions, val_transactions, k)
    mlflow.log_metric("train_map_at_12", train_map)
    # val loss
    logger.info("Recommending for validation candidates")
    val_predictions = _predict(model, val_candidates, k)
    val_map = _calculate_map(val_predictions, val_transactions, k)
    mlflow.log_metric("val_map_at_12", val_map)


def train_optuna_model(
    train_candidates: pd.DataFrame,
    val_candidates: pd.DataFrame,
    val_transactions: pd.DataFrame,
    model_params: Dict,
    optuna_params: Dict,
    k: int = 12,
) -> None:
    """Trains a LightGBM model in order to find the best hyperparameters,
    (optionally) logs the model and metrics to MLflow.

    Args:
        train_candidates (pd.DataFrame): training candidates
        val_candidates (pd.DataFrame): validation candidates
        val_transactions (pd.DataFrame): validation transactions
        model_params (Dict): base parameters for LightGBM model
        optuna_params (Dict): parameters for Optuna study search
        k (int, optional): top k recommended items. Defaults to 12.
    """
    logger.info(f"Train positive rate: {train_candidates.label.mean()}")
    train_data_size = len(train_candidates)
    features = [
        col
        for col in train_candidates.columns
        if col not in ["label", "customer_id", "article_id"]
    ]
    cat_features = train_candidates.select_dtypes(include="category").columns.to_list()
    logger.info(f"Categorical features: {cat_features}")

    # train dataset (if ranking then groups)
    train_group = (
        _prepare_groups_for_ranking(train_candidates)
        if model_params["objective"] == "lambdarank"
        else None
    )
    train_dataset = _prepare_lgb_dataset(
        train_candidates, "label", features, cat_features, train_group
    )
    # val dataset (if ranking then groups)
    val_group = (
        _prepare_groups_for_ranking(val_candidates)
        if model_params["objective"] == "lambdarank"
        else None
    )
    val_dataset = _prepare_lgb_dataset(
        val_candidates, "label", features, cat_features, val_group
    )

    mlflc = MLflowCallback(
        # tracking_uri='mlruns',
        metric_name="val_map_at_12",
        create_experiment=True,
    )

    # optuna objective
    @mlflc.track_in_mlflow()
    def objective(trial):
        model_params["n_estimators"] = trial.suggest_int("n_estimators", 2, 50)
        model_params["max_depth"] = trial.suggest_int("max_depth", 1, 20)
        model_params["num_leaves"] = trial.suggest_int("num_leaves", 2, 256)
        model_params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)

        mlflow.lightgbm.autolog(silent=True, log_models=optuna_params["log_models"])
        logger.info(
            f'Starting training model for objective: {model_params["objective"]}'
        )
        model = lgb.train(
            model_params,
            train_dataset,
            valid_sets=[train_dataset, val_dataset],
            valid_names=["train", "valid"],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(stopping_rounds=10)],
        )
        # train_data_size
        mlflow.log_param("training_dataset_size", train_data_size)
        # train loss
        logger.info("Recommending for training candidates")
        train_predictions = _predict(model, train_candidates, k)
        train_map = _calculate_map(train_predictions, val_transactions, k)
        mlflow.log_metric("train_map_at_12", train_map)
        # val loss
        logger.info("Recommending for validation candidates")
        val_predictions = _predict(model, val_candidates, k)
        val_map = _calculate_map(val_predictions, val_transactions, k)
        mlflow.log_metric("val_map_at_12", val_map)
        return val_map

    mlflow.end_run()
    study = optuna.create_study(
        study_name=optuna_params["study_name"], direction=optuna_params["direction"]
    )
    study.optimize(objective, n_trials=optuna_params["n_trials"], callbacks=[mlflc])
    logger.info(f"Best parameters from Optuna's study: {study.best_params}")
