from typing import List, Union
from gid_ml_framework.image_embeddings.data.hm_data import HMDataLoader
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from itertools import chain
import logging


log = logging.getLogger(__name__)

def _stack_predictions(predictions: List, emb_size: Union[int, str]) -> pd.DataFrame:

    out_emb, out_labels = list(), list()
    for emb, labels in predictions:
        out_emb.append(emb)
        out_labels.append(labels)
    article_ids = [article_id.split('.')[0] for article_id in chain(*out_labels)]
    column_names = [f'emb_{i+1}' for i in range(int(emb_size))]
    embeddings = torch.cat(out_emb).numpy()
    return pd.DataFrame(data=embeddings, index=article_ids, columns=column_names)

def calculate_image_embeddings(
    run_id: str,
    img_dir: str,
    batch_size: int) -> None:

    client = MlflowClient()
    run = client.get_run(run_id)
    log.info("run_id: {}".format(run.info.run_id))
    log.info("params: {}".format(run.data.params))
    log.info("status: {}".format(run.info.status))
    logged_model_uri = f'runs:/{run_id}/model'
    loaded_model = mlflow.pytorch.load_model(logged_model_uri)

    hm_dataloader = HMDataLoader(img_dir, batch_size)

    trainer = pl.Trainer(max_epochs=1, logger=False)
    predictions = trainer.predict(loaded_model, dataloaders=hm_dataloader)

    df_embeddings = _stack_predictions(predictions, run.data.params['embedding_size'])
    return df_embeddings
