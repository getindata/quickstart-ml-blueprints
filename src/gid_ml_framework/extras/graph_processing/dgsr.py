"""
Auxiliary functions for DGSR graph model
"""
import os
from typing import Any, List, Tuple
from xmlrpc.client import Boolean

import _pickle as cPickle
import dgl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SubGraphsDataset(Dataset):
    """Torch dataset for subgraphs objects"""

    def __init__(self, root_dir, loader):
        self.root = root_dir
        self.loader = loader
        self.dir_list = load_data(root_dir)
        self.size = len(self.dir_list)

    def __getitem__(self, index):
        dir_ = self.dir_list[index]
        data = self.loader(dir_)
        return data

    def __len__(self):
        return self.size


def pickle_loader(path: str) -> Any:
    pickle_object = cPickle.load(open(path, "rb"))
    return pickle_object


def select(all_items: List, user_items: List) -> np.array:
    set_diff = np.setdiff1d(all_items, user_items, assume_unique=True)
    return set_diff


def user_neg(data: pd.DataFrame, item_num: int) -> pd.DataFrame:
    """ "Generate negative items sample for all users"""
    all_items = range(item_num)
    data = data.groupby("user_id", observed=True)["item_id"].unique()
    data = data.apply(lambda x: select(all_items, x))
    return data


def neg_generate(user: int, data_neg: pd.DataFrame, neg_num: int = 100) -> np.array:
    """Sample items from all negative samples"""
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[u], neg_num, replace=False)
    return neg


def collate(data: pd.DataFrame) -> Tuple:
    """Collate function for torch subgraphs dataset"""
    user, graph, last_item, label = ([], [], [], [])
    for row in data:
        user.append(row[0])
        graph.append(row[1])
        last_item.append(row[2])
        label.append(row[3])
    return (
        torch.Tensor(user).long(),
        dgl.batch_hetero(graph),
        torch.Tensor(last_item).long(),
        torch.Tensor(label).long(),
    )


def load_data(data_path: str) -> List:
    """Generate list of all files in a dir"""
    data_dir = []
    dir_list = os.listdir(data_path)
    dir_list.sort()
    for filename in dir_list:
        for fil in os.listdir(os.path.join(data_path, filename)):
            data_dir.append(os.path.join(os.path.join(data_path, filename), fil))
    return data_dir


def collate_test(data: pd.DataFrame, user_neg: pd.DataFrame):
    """Collate function for torch test subgraphs dataset. Includes negative samples."""
    user_alis, graph, last_item, label, user, length = ([], [], [], [], [], [])
    for row in data:
        user_alis.append(row[0])
        graph.append(row[1])
        last_item.append(row[2])
        label.append(row[3])
        user.append(row[4])
        length.append(row[5])
    return (
        torch.Tensor(user_alis).long(),
        dgl.batch_hetero(graph),
        torch.Tensor(last_item).long(),
        torch.Tensor(label).long(),
        torch.Tensor(length).long(),
        torch.Tensor(neg_generate(user, user_neg)).long(),
    )


def trans_to_cuda(variable: Any) -> Any:
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def eval_metric(all_top: List, all_label: List, random_rank: Boolean = True) -> Tuple:
    """Calculates evaluation metrics based on scores"""
    recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = [], [], [], [], [], []
    data_l = np.zeros((100, 7))
    for index in range(len(all_top)):
        if random_rank:
            prediction = (-all_top[index]).argsort(1).argsort(1)
            predictions = prediction[:, 0]
            for rank in predictions:
                if rank < 20:
                    ndgg20.append(1 / np.log2(rank + 2))
                    recall20.append(1)
                else:
                    ndgg20.append(0)
                    recall20.append(0)
                if rank < 10:
                    ndgg10.append(1 / np.log2(rank + 2))
                    recall10.append(1)
                else:
                    ndgg10.append(0)
                    recall10.append(0)
                if rank < 5:
                    ndgg5.append(1 / np.log2(rank + 2))
                    recall5.append(1)
                else:
                    ndgg5.append(0)
                    recall5.append(0)
        else:
            for top_, target in zip(all_top[index], all_label[index]):
                recall20.append(np.isin(target, top_))
                recall10.append(np.isin(target, top_[0:10]))
                recall5.append(np.isin(target, top_[0:5]))
                if len(np.where(top_ == target)[0]) == 0:
                    ndgg20.append(0)
                else:
                    ndgg20.append(1 / np.log2(np.where(top_ == target)[0][0] + 2))
                if len(np.where(top_ == target)[0]) == 0:
                    ndgg10.append(0)
                else:
                    ndgg10.append(1 / np.log2(np.where(top_ == target)[0][0] + 2))
                if len(np.where(top_ == target)[0]) == 0:
                    ndgg5.append(0)
                else:
                    ndgg5.append(1 / np.log2(np.where(top_ == target)[0][0] + 2))
    return (
        np.mean(recall5),
        np.mean(recall10),
        np.mean(recall20),
        np.mean(ndgg5),
        np.mean(ndgg10),
        np.mean(ndgg20),
        pd.DataFrame(
            data_l, columns=["r5", "r10", "r20", "n5", "n10", "n20", "number"]
        ),
    )
