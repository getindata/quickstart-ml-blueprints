import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
from ...helpers.metrics import map_at_k


logger = logging.getLogger(__name__)

def _prepare_groups_for_ranking(candidates: pd.DataFrame) -> pd.DataFrame:
    candidates_group = candidates[['customer_id', 'article_id']]
    candidates_group = candidates_group.groupby(['customer_id']).size().values
    return candidates_group

def _prepare_lgb_dataset(
    candidates: pd.DataFrame,
    label: str,
    features: List[str],
    cat_features: List[str],
    group: pd.DataFrame = None) -> lgb.Dataset:
    if group is not None:
        lgb_dataset = lgb.Dataset(
            data=candidates[features],
            label=candidates[label],
            group=group,
            feature_name=features,
            categorical_feature=cat_features
        )
    else:
        lgb_dataset = lgb.Dataset(
            data=candidates[features],
            label=candidates[label],
            feature_name=features,
            categorical_feature=cat_features
        )
    return lgb_dataset

def train_val_split(candidates: pd.DataFrame, val_size: float = 0.15) -> Tuple[pd.DataFrame]:
    df_split = candidates.groupby(['customer_id'])['label'].max().reset_index()
    train_candidates, val_candidates = train_test_split(df_split, test_size=val_size, random_state=42, stratify=df_split['label'])
    train_candidates = candidates[candidates['customer_id'].isin(train_candidates['customer_id'].unique())]
    val_candidates = candidates[candidates['customer_id'].isin(val_candidates['customer_id'].unique())]
    logger.info(f'Train candidates shape: {train_candidates.shape}, \nval candidates shape{val_candidates.shape}')
    return train_candidates, val_candidates

def _predict(model: lgb.Booster, candidates: pd.DataFrame) -> pd.DataFrame:
    candidates_temp = candidates.copy()
    candidates_temp['prob'] = model.predict(candidates_temp.drop(['customer_id', 'article_id', 'label'], axis=1))
    pred_lgb = candidates_temp[['customer_id', 'article_id', 'prob']].sort_values(by=['customer_id', 'prob'], ascending=False).reset_index(drop=True)
    pred_lgb = pred_lgb.groupby(['customer_id']).head(12)
    return pred_lgb.groupby(['customer_id'])['article_id'].apply(list).reset_index()

def _calculate_map(predictions: pd.DataFrame, val_transactions: pd.DataFrame) -> float:
    df_map = (
        val_transactions
            .groupby(['customer_id'])['article_id']
            .apply(list)
            .reset_index()
            .merge(predictions, on='customer_id', how='inner')
    )
    df_map.columns = ['customer_id', 'y_true', 'y_pred']
    return map_at_k(df_map['y_true'], df_map['y_pred'], k=12)

def train_model(train_candidates: pd.DataFrame, val_candidates: pd.DataFrame, model_params: Dict, val_transactions: pd.DataFrame) -> None:
    logger.info(f'Train positive rate: {train_candidates.label.mean()}')
    features = [col for col in train_candidates.columns if col not in ['label', 'customer_id', 'article_id']]
    cat_features = train_candidates.select_dtypes(include='category').columns.to_list()
    logger.info(f'Categorical features: {cat_features}')
    if model_params['objective']=='lambdarank':
        # train dataset
        train_group = _prepare_groups_for_ranking(train_candidates)
        train_dataset = _prepare_lgb_dataset(train_candidates, 'label', features, cat_features, train_group)
        # val dataset
        val_group = _prepare_groups_for_ranking(val_candidates)
        val_dataset = _prepare_lgb_dataset(val_candidates, 'label', features, cat_features, val_group)
    elif model_params['objective']=='binary':
        # train dataset
        train_dataset = _prepare_lgb_dataset(train_candidates, 'label', features, cat_features)
        # val dataset
        val_dataset = _prepare_lgb_dataset(val_candidates, 'label', features, cat_features)
    mlflow.lightgbm.autolog(silent=True)
    logger.info(f'Starting training model for objective: {model_params["objective"]}')
    model = lgb.train(
        model_params,
        train_dataset,
        valid_sets=[train_dataset, val_dataset],
        valid_names=['train', 'valid'],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    # train loss
    logger.info('Recommending for training candidates')
    train_predictions = _predict(model, train_candidates)
    train_map = _calculate_map(train_predictions, val_transactions)
    mlflow.log_metric('train_map_at_12', train_map)
    # val loss
    logger.info('Recommending for validation candidates')
    val_predictions = _predict(model, val_candidates)
    val_map = _calculate_map(val_predictions, val_transactions)
    mlflow.log_metric('val_map_at_12', val_map)
