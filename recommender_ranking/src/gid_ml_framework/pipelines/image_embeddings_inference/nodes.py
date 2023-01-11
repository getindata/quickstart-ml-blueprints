import logging
from itertools import chain
from typing import Dict, List, Union

import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
import torch

from gid_ml_framework.image_embeddings.data.hm_data import HMDataLoader

logger = logging.getLogger(__name__)


def _stack_predictions(predictions: List, emb_size: Union[int, str]) -> pd.DataFrame:
    """Stacking predictions from list, and saving them as pd.DataFrame

    Args:
        predictions (List): list containing tuples of embeddings, labels. Each tuple is the size of batch size
        emb_size (Union[int, str]): embedding size

    Returns:
        pd.DataFrame: embeddings
    """
    out_emb, out_labels = list(), list()
    for emb, labels in predictions:
        out_emb.append(emb)
        out_labels.append(labels)
    article_ids = [article_id.split(".")[0] for article_id in chain(*out_labels)]
    column_names = [f"img_emb_{i+1}" for i in range(int(emb_size))]
    embeddings = torch.cat(out_emb).numpy()
    return pd.DataFrame(data=embeddings, index=article_ids, columns=column_names)


def calculate_image_embeddings(
    model_uri: str,
    img_dir: str,
    platform: str,
    batch_size: int,
    training_metadata: Dict,
) -> pd.DataFrame:
    """Generates image embeddings, given a trained model from MLflow model URI.

    Args:
        model_uri (str): MLflow model URI
        img_dir (str): directory with images to generate embeddings
        platform (str): local or gcp
        batch_size (int): batch size
        training_metadata (Dict): image autoencoder training metadata

    Returns:
        pd.DataFrame: image embeddings
    """
    logger.info(f"Loading model from: {model_uri=}")
    loaded_model = mlflow.pytorch.load_model(model_uri)
    logger.info(f"Loading data from {img_dir=} on {platform=}")
    hm_dataloader = HMDataLoader(img_dir, batch_size, platform=platform)
    trainer = pl.Trainer(max_epochs=1, logger=False)
    logger.info("Starting generating predictions")
    predictions = trainer.predict(loaded_model, dataloaders=hm_dataloader)
    embedding_size = loaded_model.encoder.embedding_size
    logger.info(f"Stacking all predictions, with {embedding_size=}")
    df_embeddings = _stack_predictions(predictions, embedding_size)
    return df_embeddings
