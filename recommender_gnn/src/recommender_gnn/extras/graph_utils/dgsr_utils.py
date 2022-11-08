"""
Auxiliary functions for DGSR graph model
"""
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import dgl
import numpy as np
import pandas as pd
import torch
from dgl import DGLHeteroGraph
from google.cloud import storage
from torch.utils.data import Dataset


def load_graphs_python(graph_dir: str) -> List:
    """Load heterpgraphs from a given path using only Python functions instead of dgl C implementation"""
    with open(graph_dir, "rb") as f:
        data = pickle.load(f)
    return data


def save_graphs_python(save_filepath: str, graphs_dict: Dict[str, List]) -> None:
    """Save heterographs into file using only Python functions instead of dgl C implementation"""
    path_object = Path(save_filepath)
    path_parts = path_object.parts
    if path_parts[0] == "gs:":
        save_to_bucket(path_parts, graphs_dict)
    else:
        with open(save_filepath, "wb") as file:
            pickle.dump(graphs_dict, file, protocol=-1)


def create_graphs_list(
    graph: dgl.DGLGraph, graph_dict: Dict
) -> List[Union[dgl.DGLGraph, Dict]]:
    if graph_dict is None:
        graph_dict = {}
    if isinstance(graph, DGLHeteroGraph):
        graph = [graph]
        graph_dict = [graph_dict]
    assert all(
        [type(g) == DGLHeteroGraph for g in graph]
    ), "Invalid DGLHeteroGraph in graph argument"
    gdata_list = [[g, graph_dict[i]] for i, g in enumerate(graph)]
    return gdata_list


class SubGraphsDataset(Dataset):
    """Torch dataset for subgraphs objects"""

    def __init__(self, root_dir, loader):
        self.root = root_dir
        graphs_collection_path = os.path.join(root_dir, "graphs.bin")
        self.graphs_collection = loader(graphs_collection_path)
        self.keys = list(self.graphs_collection.keys())
        self.size = len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        data = self.graphs_collection.get(key, None)
        return data, key

    def __len__(self):
        return self.size


def save_to_bucket(path_parts: List[str], gdata_list: List) -> None:
    """Save graph files as pickle to Google Cloud Storage bucket"""
    storage_client = storage.Client()
    bucket_name = path_parts[1]
    bucket = storage_client.bucket(bucket_name)
    blob_name = Path(*path_parts[2:])
    blob = bucket.blob(str(blob_name))
    pickle_out = pickle.dumps(gdata_list, protocol=-1)
    blob.upload_from_string(pickle_out)


def select(all_items: List, user_items: List) -> np.array:
    set_diff = np.setdiff1d(all_items, user_items, assume_unique=True)
    return set_diff


def user_neg(data: pd.DataFrame, item_num: int) -> pd.DataFrame:
    """Generate negative items sample for all users"""
    all_items = range(item_num)
    data = data.groupby("user_id")["item_id"].apply(lambda x: select(all_items, x))
    return data


def neg_generate(user: List, data_neg: pd.DataFrame, item_num: int) -> np.array:
    """Sample items from all negative samples"""
    neg_num = 100
    max_users = 50
    if item_num > neg_num + max_users:
        replace = False
    else:
        replace = True
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[u.item()], neg_num, replace=replace)
    return neg


def collate(data: pd.DataFrame) -> Tuple:
    """Collate function for torch subgraphs dataset"""
    user, user_l, graph, label, last_item = ([], [], [], [], [])
    # Get user ids from path
    original_user_ids = [int(Path(row[1]).stem.split("_")[0]) for row in data]
    graphs = [row[0][0] for row in data]
    for row in graphs:
        user.append(row[1]["user"])
        user_l.append(row[1]["u_alias"])
        graph.append(row[0])
        label.append(row[1]["target"])
        last_item.append(row[1]["last_alias"])
    return (
        torch.tensor(user_l).long(),
        dgl.batch(graph),
        torch.tensor(label).long(),
        torch.tensor(last_item).long(),
        torch.tensor(original_user_ids).long(),
    )


def collate_test(data: pd.DataFrame, user_neg: pd.DataFrame, item_num: int):
    """Collate function for torch test subgraphs dataset. Includes negative samples."""
    user, graph, label, last_item = ([], [], [], [])
    data = [row[0][0] for row in data]
    for row in data:
        user.append(row[1]["u_alias"])
        graph.append(row[0])
        label.append(row[1]["target"])
        last_item.append(row[1]["last_alias"])
    return (
        torch.tensor(user).long(),
        dgl.batch(graph),
        torch.tensor(label).long(),
        torch.tensor(last_item).long(),
        torch.Tensor(neg_generate(user, user_neg, item_num)).long(),
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


def trans_to_cuda(variable: Any) -> Any:
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def graph_user(
    bg: dgl.DGLGraph, user_index: int, user_embedding: torch.tensor
) -> torch.tensor:
    """Generates new user embedding after single forward pass"""
    b_user_size = bg.batch_num_nodes("user")
    tmp = torch.roll(torch.cumsum(b_user_size, 0), 1)
    tmp[0] = 0
    new_user_index = tmp + user_index
    return user_embedding[new_user_index]


def graph_item(
    bg: dgl.DGLGraph, last_index: int, item_embedding: torch.tensor
) -> torch.tensor:
    """Generates new item embedding after single forward pass"""
    b_item_size = bg.batch_num_nodes("item")
    tmp = torch.roll(torch.cumsum(b_item_size, 0), 1)
    tmp[0] = 0
    new_item_index = tmp + last_index
    return item_embedding[new_item_index]


def eval_metric(all_top: List) -> Tuple[float]:
    """Calculates evaluation metrics based on scores"""
    recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = ([], [], [], [], [], [])
    for index in range(len(all_top)):
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
    return (
        np.mean(recall5),
        np.mean(recall10),
        np.mean(recall20),
        np.mean(ndgg5),
        np.mean(ndgg10),
        np.mean(ndgg20),
    )


def mkdir_if_not_exist(file_name: str) -> None:
    dir_name = os.path.dirname(file_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


class Logger(object):
    """Original paper logger for training"""

    def __init__(self, file_path: str) -> None:
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        """This flush method is needed for python 3 compatibility. This handles the flush command by doing nothing.
        You might want to specify some extra behavior here.
        """
        pass
