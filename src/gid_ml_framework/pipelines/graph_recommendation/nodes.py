import datetime
import logging
from operator import itemgetter
from typing import Dict, Tuple
from xmlrpc.client import Boolean

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from gid_ml_framework.extras.graph_utils.dgsr_utils import (
    SubGraphsDataset,
    collate,
    collate_test,
    eval_metric,
)
from gid_ml_framework.gnn_models.recommendation.dgsr import DGSR

pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)


def _get_data_stats(
    transactions: pd.DataFrame,
    train_set: SubGraphsDataset,
    test_set: SubGraphsDataset,
) -> Tuple[int]:

    user = transactions["user_id"].unique()
    item = transactions["item_id"].unique()
    user_num = len(user)
    item_num = len(item)

    logger.info("Train set size:", train_set.size)
    logger.info("Test set size:", test_set.size)
    logger.info("Number of all unique users:", user_num)
    logger.info("Number of all unique items:", item_num)
    return user_num, item_num


def _get_loaders(
    train_set: SubGraphsDataset,
    val_set: SubGraphsDataset,
    test_set: SubGraphsDataset,
    negative_samples: pd.DataFrame,
    train_params: Dict,
    validate: Boolean,
) -> Tuple[DataLoader]:
    """Creates torch DataLoader from given datasets. Collates negative samples with test and val sets."""
    batch_size, validate = itemgetter("batch_size", validate)(train_params)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        collate_fn=lambda x: collate_test(x, negative_samples),
        pin_memory=True,
        num_workers=0,
    )
    if validate:
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=batch_size,
            collate_fn=lambda x: collate_test(x, negative_samples),
            pin_memory=True,
            num_workers=0,
        )
    else:
        val_loader = test_loader
    return train_loader, val_loader, test_loader


def _get_model(device: str, model_params: Dict, data_stats: Tuple[int]) -> nn.Module:
    user_num, item_num = data_stats
    model = DGSR(
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
    params_names = ["lr", "l2", "epoch", "validate"]
    params = itemgetter(params_names)(train_params)
    return params


def train_model(
    train_set: SubGraphsDataset,
    val_set: SubGraphsDataset,
    test_set: SubGraphsDataset,
    transactions: pd.DataFrame,
    negative_samples: pd.DataFrame,
    model_params: Dict,
    train_params: Dict,
    k: int = 20,
) -> None:
    """Trains a GNN recommendation model, logs the model and metrics to MLflow.

    Args:
        train_set (SubGraphsDataset): training subset of data
        val_set (SubGraphsDataset): validation subset of data
        test_set (SubGraphsDataset): test subset of data
        model_params (Dict): parameters for chosen GNN model
        validate (Boolean): flag indicating if model should be scored on validation dataset
        k (int, optional): top k recommended items. Defaults to 20.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = _get_loaders(
        train_set, val_set, test_set, negative_samples, train_params
    )
    data_stats = _get_data_stats(transactions, train_set, test_set)
    model = _get_model(device, model_params, data_stats)
    lr, l2, epoch, validate = _unpack_train_params(train_params)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    loss_func = nn.CrossEntropyLoss()
    best_result = [0, 0, 0, 0, 0, 0]  # hit5,hit10,hit20,mrr5,mrr10,mrr20
    best_epoch = [0, 0, 0, 0, 0, 0]
    stop_num = 0
    for epoch in range(epoch):
        stop = True
        epoch_loss = 0
        iter = 0
        logger.info("start training: ", datetime.datetime.now())
        model.train()
        for user, batch_graph, label, last_item in train_loader:
            iter += 1
            score = model(
                batch_graph.to(device),
                user.to(device),
                last_item.to(device),
                is_training=True,
            )
            loss = loss_func(score, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if iter % 400 == 0:
                logger.info(
                    "Iter {}, loss {:.4f}".format(iter, epoch_loss / iter),
                    datetime.datetime.now(),
                )
        epoch_loss /= iter
        model.eval()
        logger.info(
            "Epoch {}, loss {:.4f}".format(epoch, epoch_loss),
            "=============================================",
        )

        # val
        if validate:
            logger.info("start validation: ", datetime.datetime.now())
            val_loss_all, top_val = [], []
            with torch.no_grad:
                for user, batch_graph, label, last_item, neg_tar in val_loader:
                    score, top = model(
                        batch_graph.to(device),
                        user.to(device),
                        last_item.to(device),
                        neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device),
                        is_training=False,
                    )
                    val_loss = loss_func(score, label.cuda())
                    val_loss_all.append(val_loss.append(val_loss.item()))
                    top_val.append(top.detach().cpu().numpy())
                recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(
                    top_val
                )
                logger.info(
                    "train_loss:%.4f\tval_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f"
                    "\tNDGG10@10:%.4f\tNDGG@20:%.4f"
                    % (
                        epoch_loss,
                        np.mean(val_loss_all),
                        recall5,
                        recall10,
                        recall20,
                        ndgg5,
                        ndgg10,
                        ndgg20,
                    )
                )

        # test
        logger.info("start predicting: ", datetime.datetime.now())
        all_top, all_label = [], []
        iter = 0
        all_loss = []
        with torch.no_grad():
            for user, batch_graph, label, last_item, neg_tar in test_loader:
                iter += 1
                score, top = model(
                    batch_graph.to(device),
                    user.to(device),
                    last_item.to(device),
                    neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device),
                    is_training=False,
                )
                test_loss = loss_func(score, label.cuda())
                all_loss.append(test_loss.item())
                all_top.append(top.detach().cpu().numpy())
                all_label.append(label.numpy())
                if iter % 200 == 0:
                    logger.info(
                        "Iter {}, test_loss {:.4f}".format(iter, np.mean(all_loss)),
                        datetime.datetime.now(),
                    )
            recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(all_top)
            if recall5 > best_result[0]:
                best_result[0] = recall5
                best_epoch[0] = epoch
                stop = False
            if recall10 > best_result[1]:
                best_result[1] = recall10
                best_epoch[1] = epoch
                stop = False
            if recall20 > best_result[2]:
                best_result[2] = recall20
                best_epoch[2] = epoch
                stop = False
                # ------select Mrr------------------
            if ndgg5 > best_result[3]:
                best_result[3] = ndgg5
                best_epoch[3] = epoch
                stop = False
            if ndgg10 > best_result[4]:
                best_result[4] = ndgg10
                best_epoch[4] = epoch
                stop = False
            if ndgg20 > best_result[5]:
                best_result[5] = ndgg20
                best_epoch[5] = epoch
                stop = False
            if stop:
                stop_num += 1
            else:
                stop_num = 0
            logger.info(
                "train_loss:%.4f\ttest_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f"
                "\tNDGG10@10:%.4f\tNDGG@20:%.4f\tEpoch:%d,%d,%d,%d,%d,%d"
                % (
                    epoch_loss,
                    np.mean(all_loss),
                    best_result[0],
                    best_result[1],
                    best_result[2],
                    best_result[3],
                    best_result[4],
                    best_result[5],
                    best_epoch[0],
                    best_epoch[1],
                    best_epoch[2],
                    best_epoch[3],
                    best_epoch[4],
                    best_epoch[5],
                )
            )
