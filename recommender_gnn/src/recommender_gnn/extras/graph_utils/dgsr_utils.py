"""
Auxiliary functions for DGSR graph model
"""
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import dgl
import numpy as np
import pandas as pd
import torch
from dgl import DGLHeteroGraph
from google.cloud import storage
from google.cloud.storage import Blob
from torch.utils.data import Dataset


def load_graphs_python(load_filepath: str) -> Dict[str, List]:
    """Load heterpgraphs from a given path using only Python functions instead of dgl C implementation"""
    path_parts = split_path(load_filepath)
    if path_parts[0] == "gs:":
        graphs_dict = load_graphs_from_bucket(path_parts)
    else:
        with open(load_filepath, "rb") as f:
            graphs_dict = pickle.load(f)
    return graphs_dict


def save_graphs_python(save_filepath: str, graphs_dict: Dict[str, List]) -> None:
    """Save heterographs into file using only Python functions instead of dgl C implementation"""
    path_parts = split_path(save_filepath)
    if path_parts[0] == "gs:":
        save_graphs_to_bucket(path_parts, graphs_dict)
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

    def __init__(self, root_dir, loader, extension="bin"):
        self.root = root_dir
        graphs_collection_path = os.path.join(root_dir, f"graphs.{extension}")
        self.graphs_collection = loader(graphs_collection_path)
        self.keys = list(self.graphs_collection.keys())
        self.size = len(self.keys)

    def get_graphs_collection(self):
        return self.graphs_collection

    def __getitem__(self, index):
        key = self.keys[index]
        data = self.graphs_collection.get(key, None)
        return data, key

    def __len__(self):
        return self.size


def split_path(file_path: str) -> List[str]:
    path_object = Path(file_path)
    path_parts = path_object.parts
    return path_parts


def connect_to_gcs_blob(path_parts: List[str]) -> Blob:
    """Connects to target file in gcs and returns its blob object"""
    storage_client = storage.Client()
    bucket_name = path_parts[1]
    bucket = storage_client.bucket(bucket_name)
    blob_name = Path(*path_parts[2:])
    blob = bucket.blob(str(blob_name))
    return blob


def save_graphs_to_bucket(path_parts: List[str], gdata_list: List) -> None:
    """Save graph files as pickle to Google Cloud Storage bucket"""
    blob = connect_to_gcs_blob(path_parts)
    pickle_out = pickle.dumps(gdata_list, protocol=-1)
    blob.upload_from_string(pickle_out)


def load_graphs_from_bucket(path_parts: List[str]) -> Dict[str, List]:
    """Load graph files as pickle from Google Cloud Storage bucket"""
    blob = connect_to_gcs_blob(path_parts)
    pickle_in = blob.download_as_string()
    graphs_dict = pickle.loads(pickle_in)
    return graphs_dict


def select(all_items: List, user_items: List) -> np.array:
    set_diff = np.setdiff1d(all_items, user_items, assume_unique=True)
    return set_diff


def user_neg(data: pd.DataFrame, item_num: int) -> pd.DataFrame:
    """Generate negative items sample for all users"""
    all_items = range(item_num)
    data = data.groupby("user_id")["item_id"].apply(lambda x: select(all_items, x))
    return data


def generate_negative_samples(
    user: List, data_neg: pd.DataFrame, item_num: int
) -> np.array:
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
        torch.Tensor(generate_negative_samples(user, user_neg, item_num)).long(),
    )


def generate_embedding(
    batch_graph: dgl.DGLGraph, index: int, embedding: torch.tensor, node_type: str
) -> torch.tensor:
    """Generates new type (item/user) of embedding after single forward pass"""
    batch_size = batch_graph.batch_num_nodes(node_type)
    rolled_batch = torch.roll(torch.cumsum(batch_size, 0), 1)
    rolled_batch[0] = 0
    new_index = rolled_batch + index
    return embedding[new_index]


def eval_metric(all_top: List) -> Tuple[float]:
    """Calculates evaluation metrics (recall@x and ndgg@x) based on scores"""
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
