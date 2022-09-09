import logging
from operator import itemgetter
from typing import Dict, Tuple, Union

import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from mlflow import MlflowClient
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader

from gid_ml_framework.extras.datasets.chunks_dataset import _concat_chunks
from gid_ml_framework.extras.graph_utils.dgsr_utils import (
    SubGraphsDataset,
    collate,
    collate_test,
)
from gid_ml_framework.gnn_models.recommendation.dgsr import DGSR

pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)


def _get_data_stats(
    transactions: pd.DataFrame,
    train_set: SubGraphsDataset,
) -> Tuple[int]:

    user = transactions["user_id"].unique()
    item = transactions["item_id"].unique()
    user_num = len(user)
    item_num = len(item)

    logger.info(f"Train set size: {train_set.size}")
    logger.info(f"Number of all unique users: {user_num}")
    logger.info(f"Number of all unique items: {item_num}")
    return user_num, item_num


def _get_loaders(
    train_set: Union[SubGraphsDataset, None],
    val_set: Union[SubGraphsDataset, None],
    test_set: Union[SubGraphsDataset, None],
    negative_samples: pd.DataFrame,
    train_params: Dict,
    data_stats: Tuple[int],
) -> Tuple[DataLoader, None]:
    """Creates torch DataLoader from given datasets. Collates negative samples with test and val sets."""
    batch_size = train_params.get("batch_size")
    _, item_num = data_stats
    train_loader, val_loader, test_loader = (None, None, None)
    if train_set:
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            collate_fn=collate,
            shuffle=True,
            pin_memory=True,
            num_workers=12,
        )
    if train_set:
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            collate_fn=lambda x: collate_test(x, negative_samples, item_num),
            pin_memory=True,
            num_workers=12,
        )
    if test_set:
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=batch_size,
            collate_fn=lambda x: collate_test(x, negative_samples, item_num),
            pin_memory=True,
            num_workers=12,
        )
    return train_loader, val_loader, test_loader


def _get_model(
    device: str, model_params: Dict, train_params: Dict, data_stats: Tuple[int]
) -> nn.Module:
    user_num, item_num = data_stats
    model = DGSR(
        lr=train_params.get("lr"),
        l2=train_params.get("l2"),
        user_num=user_num,
        item_num=item_num,
        input_dim=model_params.get("hidden_size"),
        item_max_length=model_params.get("item_max_length"),
        user_max_length=model_params.get("user_max_length"),
        feat_drop=model_params.get("feat_drop_out"),
        attn_drop=model_params.get("attn_drop_out"),
        user_long=model_params.get("user_long"),
        user_short=model_params.get("user_short"),
        item_long=model_params.get("item_long"),
        item_short=model_params.get("item_short"),
        user_update=model_params.get("user_update"),
        item_update=model_params.get("item_update"),
        last_item=model_params.get("last_item"),
        layer_num=model_params.get("layer_num"),
    ).to(device)
    return model


def _unpack_train_params(train_params) -> Tuple:
    params_names = ["epochs", "validate"]
    params = itemgetter(*params_names)(train_params)
    return params


def _log_auto_logged_info(mlflow_run: mlflow.entities.Run) -> None:
    tags = {
        k: v for k, v in mlflow_run.data.tags.items() if not k.startswith("mlflow.")
    }
    artifacts = [
        f.path for f in MlflowClient().list_artifacts(mlflow_run.info.run_id, "model")
    ]
    logger.info(f"run_id: {mlflow_run.info.run_id}")
    logger.info(f"artifacts: {artifacts}")
    logger.info(f"params: {mlflow_run.data.params}")
    logger.info(f"metrics: {mlflow_run.data.metrics}")
    logger.info(f"tags: {tags}")


def train_model(
    train_set: SubGraphsDataset,
    val_set: SubGraphsDataset,
    transactions: pd.DataFrame,
    negative_samples: pd.DataFrame,
    model_params: Dict,
    train_params: Dict,
    save_model: bool = False,
    seed: int = 321,
) -> pl.LightningModule:
    """Trains a GNN recommendation model, logs the model and metrics to MLflow.

    Args:
        train_set (SubGraphsDataset): training subset of data
        val_set (SubGraphsDataset): validation subset of data
        test_set (SubGraphsDataset): test subset of data
        model_params (Dict): parameters for chosen GNN model
        train_params (Dict): parameters for training process
    """
    seed_everything(seed, True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transactions = _concat_chunks(transactions)
    data_stats = _get_data_stats(transactions, train_set)
    train_loader, val_loader, _ = _get_loaders(
        train_set, val_set, None, negative_samples, train_params, data_stats
    )
    model = _get_model(device, model_params, train_params, data_stats)
    epochs, validate = _unpack_train_params(train_params)

    if validate:
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss", min_delta=0.00001, patience=5, verbose=True, mode="min"
        )
    else:
        early_stop_callback = None

    trainer = pl.Trainer(
        max_epochs=epochs,
        devices=1,
        accelerator="auto",
        enable_checkpointing=False,
        callbacks=early_stop_callback,
    )

    mlflow.pytorch.autolog(log_models=save_model)
    if validate:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loader)
    _log_auto_logged_info(mlflow.active_run())
    return model


def test_model():
    pass


def get_predictions(predict_set: SubGraphsDataset, model) -> pd.DataFrame:
    trainer = pl.Trainer(devices=1, accelerator="auto")
    predict_dataloader, _, _ = _get_loaders(predict_set, None, None)
    predictions = trainer.predict(model, dataloaders=predict_dataloader)
    return predictions


def tune_model():
    pass
