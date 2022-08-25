import mlflow.lightgbm
import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import List
import logging


logger = logging.getLogger(__name__)

def _load_model(run_id: str) -> lgb.Booster:
    logger.info(f'Loading model from {run_id=}')
    logged_model_uri = f'runs:/{run_id}/model'
    loaded_model = mlflow.lightgbm.load_model(logged_model_uri)
    return loaded_model

def _predict(candidates: pd.DataFrame, lgb_model: lgb.Booster, k: int = 12, batch_size: int = 1_000_000) -> pd.DataFrame:
    preds = []
    candidates = candidates.copy()
    for bucket in range(0, len(candidates), batch_size):
        outputs = lgb_model.predict(candidates.iloc[bucket:bucket+batch_size].drop(['customer_id', 'article_id'], axis=1))
        preds.append(outputs)
    probs = np.concatenate(preds)
    candidates['prob'] = probs
    candidates = candidates[['customer_id', 'article_id', 'prob']]
    predictions = candidates.sort_values(by=['customer_id', 'prob'], ascending=False).reset_index(drop=True)
    predictions = predictions.groupby(['customer_id']).head(k)
    predictions = predictions.groupby(['customer_id'])['article_id'].apply(list).reset_index()
    return predictions

def _weight_blend(predictions: pd.DataFrame, prediction_cols: List[str], k: int = 12) -> pd.DataFrame:
    # changed a bit from https://www.kaggle.com/code/titericz/h-m-ensembling-how-to 
    # Create a list of all model predictions
    recs = list()
    for pred_col in prediction_cols:
        recs.append(predictions[pred_col])
    
    # Create a dictionary of items recommended.
    # Assign a weight according the order of appearance (TODO and multiply by global weights)
    res = dict()
    for m in range(len(recs)):
        for n, article_id in enumerate(recs[m]):
            res[article_id] = res.get(article_id, 0) + 1/(n+1)

    # Sort dictionary by item weights
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())
    # kaggle submission format + top 12 only
    res = ' '.join(res[:k])
    return res

def _ensemble_predictions(predictions: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    prediction_cols = [col for col in predictions if col.startswith('prediction_')]
    predictions['prediction'] = predictions.apply(_weight_blend, prediction_cols=prediction_cols, k=k, axis=1)
    predictions = predictions.drop(prediction_cols, axis=1)
    return predictions

def _predict_from_multiple_models(candidates: pd.DataFrame, models: List[str], k: int = 12):
    preds = []
    for i, run_id in enumerate(models):
        model = _load_model(run_id)
        predictions = _predict(candidates, model)
        predictions.columns = ['customer_id', f'prediction_{i}']
        logger.info(f'Predictions shape: {predictions.shape}')
        preds.append(predictions)
    predictions = pd.concat([df.set_index('customer_id') for df in preds], axis=1).reset_index()
    logger.info(f'Concatenated predictions shape: {predictions.shape}')
    predictions = _ensemble_predictions(predictions, k)
    return predictions

def _predict_from_single_model(candidates: pd.DataFrame, model_str: str, k: int = 12):
    model = _load_model(model_str)
    predictions = _predict(candidates, model, k)
    predictions.columns = ['customer_id', 'prediction']
    # kaggle submission format
    predictions['prediction'] = predictions['prediction'].apply(lambda x: ' '.join(x))
    return predictions

def generate_predictions(candidates: pd.DataFrame, models: List[str], k: int = 12):
    candidates = candidates.drop(['label'], axis=1)
    # multiple models ensemble
    if len(models)>1: # isinstance(obj, list/str)?
        return _predict_from_multiple_models(candidates, models, k)
    # single model
    return _predict_from_single_model(candidates, models[0], k)
