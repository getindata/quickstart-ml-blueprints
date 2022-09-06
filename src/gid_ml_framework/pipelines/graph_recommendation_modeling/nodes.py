import logging
from typing import Dict, List, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.sampling import select_topk
from joblib import Parallel, delayed

from gid_ml_framework.extras.datasets.chunks_dataset import _concat_chunks
from gid_ml_framework.extras.graph_processing.dgsr import user_neg

pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)


def _transactions_order(data: pd.DataFrame, colname: str) -> pd.DataFrame:
    data = data.sort_values(["time"], kind="mergesort")
    data[colname] = range(len(data))
    return data


def _refine_time(data: pd.DataFrame) -> pd.DataFrame:
    """Adds time gaps to events with same timestamp to introduce order."""
    data = data.sort_values(["time"], kind="mergesort")
    time_seq = data.loc[:, "time"].values
    time_gap = 1
    for i, _ in enumerate(time_seq[0:-1]):
        if time_seq[i] == time_seq[i + 1] or time_seq[i] > time_seq[i + 1]:
            time_seq[i + 1] = time_seq[i + 1] + time_gap
            time_gap += 1
    data.loc[:, "time"] = time_seq
    return data


def _simple_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby("user_id").apply(_refine_time).reset_index(drop=True)
    df.loc[:, "time"] = df.loc[:, "time"].astype("int64")
    return df


def generate_graph_dgsr(df: pd.DataFrame) -> dgl.DGLGraph:
    """Generates graph from whole dataset.

    Args:
        df (pd.DataFrame): dataframe with mapped transactions

    Returns:
        dgl.DGLGraph: graph object created from transaction in input dataframe
    """
    df = _concat_chunks(df)
    data = _simple_preprocess(df)
    data = data.groupby("user_id").apply(_refine_time).reset_index(drop=True)
    data = (
        data.groupby("user_id")
        .apply(lambda x: _transactions_order(x, "order"))
        .reset_index(drop=True)
    )
    data = (
        data.groupby("item_id")
        .apply(lambda x: _transactions_order(x, "u_order"))
        .reset_index(drop=True)
    )
    user = data.loc[:, "user_id"].values
    item = data.loc[:, "item_id"].values
    time = data.loc[:, "time"].values
    # Heterogeneous bidirectional graph with interactions between users and items with information about timestamps
    graph_data = {
        ("item", "by", "user"): (torch.tensor(item), torch.tensor(user)),
        ("user", "pby", "item"): (torch.tensor(user), torch.tensor(item)),
    }
    graph = dgl.heterograph(graph_data)
    graph.edges["by"].data["time"] = torch.LongTensor(time)
    graph.edges["pby"].data["time"] = torch.LongTensor(time)
    graph.nodes["user"].data["user_id"] = torch.LongTensor(np.unique(user))
    graph.nodes["item"].data["item_id"] = torch.LongTensor(np.unique(item))
    return graph


def _generate_subgraphs(
    graph: dgl.DGLGraph,
    u_time: List,
    start_t: int,
    user: int,
    item_max_length: int,
    j: int,
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.tensor]:
    """Generates subgraph for user based on transactions timestamps."""
    sub_u_eid = (graph.edges["by"].data["time"] < u_time[j + 1]) & (
        graph.edges["by"].data["time"] >= start_t
    )
    sub_i_eid = (graph.edges["pby"].data["time"] < u_time[j + 1]) & (
        graph.edges["pby"].data["time"] >= start_t
    )
    sub_graph = dgl.edge_subgraph(
        graph, edges={"by": sub_u_eid, "pby": sub_i_eid}, relabel_nodes=False
    )
    u_temp = torch.tensor([user])
    graph_i = select_topk(
        sub_graph, item_max_length, weight="time", nodes={"user": u_temp}
    )
    return graph_i, sub_graph, u_temp


def _iterate_subgraphs(
    sub_graph: dgl.DGLGraph,
    graph_i: dgl.DGLGraph,
    k_hop: int,
    user_max_length: int,
    item_max_length: int,
    his_user: int,
) -> Tuple:
    """Iterates through subgraph to sample neighbors and create final graph"""
    i_temp = torch.unique(graph_i.edges(etype="by")[0])
    his_item = torch.unique(graph_i.edges(etype="by")[0])
    edge_i = [graph_i.edges["by"].data[dgl.NID]]
    edge_u = []
    # Iterating for k_hop neighbors
    for _ in range(k_hop - 1):
        graph_u = select_topk(
            sub_graph, user_max_length, weight="time", nodes={"item": i_temp}
        )
        u_temp = np.setdiff1d(torch.unique(graph_u.edges(etype="pby")[0]), his_user)[
            -user_max_length:
        ]
        graph_i = select_topk(
            sub_graph, item_max_length, weight="time", nodes={"user": u_temp}
        )
        his_user = torch.unique(torch.cat([torch.tensor(u_temp), his_user]))
        i_temp = np.setdiff1d(torch.unique(graph_i.edges(etype="by")[0]), his_item)
        his_item = torch.unique(torch.cat([torch.tensor(i_temp), his_item]))
        edge_i.append(graph_i.edges["by"].data[dgl.NID])
        edge_u.append(graph_u.edges["pby"].data[dgl.NID])
    all_edge_u = torch.unique(torch.cat(edge_u))
    all_edge_i = torch.unique(torch.cat(edge_i))
    # Creating final graph
    fin_graph = dgl.edge_subgraph(
        sub_graph, edges={"by": all_edge_i, "pby": all_edge_u}
    )
    return fin_graph


def _prepare_graph_dict(
    fin_graph: dgl.DGLGraph,
    user: int,
    u_seq: List,
    j: int,
) -> Dict:
    target = u_seq[j + 1]
    last_item = u_seq[j]
    u_alis = torch.where(fin_graph.nodes["user"].data["user_id"] == user)[0]
    last_alis = torch.where(fin_graph.nodes["item"].data["item_id"] == last_item)[0]
    graph_dict = {
        "user": torch.tensor([user]),
        "target": torch.tensor([target]),
        "u_alis": u_alis,
        "last_alis": last_alis,
    }
    return graph_dict


def _prepare_user_data(data: pd.DataFrame, user: int) -> Tuple:
    data_user = data[data["user_id"] == user].sort_values("time")
    data_user["time"] = data_user["time"].astype("int32")
    u_time = data_user["time"].values
    u_seq = data_user["item_id"].values
    split_point = len(u_seq) - 1
    return u_seq, u_time, split_point


def _generate_user_subgraphs(
    user: int,
    data: pd.DataFrame,
    graph: dgl.DGLGraph,
    item_max_length: int,
    user_max_length: int,
    k_hop: int = 3,
) -> Tuple:
    """Generates subgraphs of transactions for each users interaction.

    Args:
        user (int): user id from mapped transactions dataframe
        data (pd.DataFrame): dataframe with mapped transactions
        graph (dgl.DGLGraph): graph object created from whole dataset
        item_max_length (int): maximum length of item interactions for single user
        user_max_length (int): maximum number of users to sample into subgraph
        k_hop (int): number of iterations for subgraphs creation

    Returns:
        Tuple: (train_list, val_list, test_list) - each one contains subgraphs for train/val/test subsets
    """
    u_seq, u_time, split_point = _prepare_user_data(data, user)
    train_list, val_list, test_list = ([], [], [])
    # Considering only users with at least 3 transactions
    if len(u_seq) < 3:
        return train_list, val_list, test_list
    else:
        for j, _ in enumerate(u_time[0:-1]):
            if j == 0:
                continue
            if j < item_max_length:
                start_t = u_time[0]
            else:
                start_t = u_time[j - item_max_length]
            graph_i, sub_graph, his_user = _generate_subgraphs(
                graph, u_time, start_t, user, item_max_length, j
            )
            fin_graph = _iterate_subgraphs(
                sub_graph, graph_i, k_hop, user_max_length, item_max_length, his_user
            )
            graph_dict = _prepare_graph_dict(fin_graph, user, u_seq, j)
            # Train/val/test split based on order in sequence
            if j < split_point - 1:
                train_list.append([user, j, fin_graph, graph_dict])
            if j == split_point - 1 - 1:
                val_list.append([user, j, fin_graph, graph_dict])
            if j == split_point - 1:
                test_list.append([user, j, fin_graph, graph_dict])
    return train_list, val_list, test_list


def _generate_model_input(
    data,
    graph,
    item_max_length,
    user_max_length,
    job=10,
    k_hop=3,
):
    """Wrapper for generating subgraphs for each user"""
    user = data.loc[:, "user_id"].unique()
    sample_lists = Parallel(n_jobs=job)(
        delayed(
            lambda u: _generate_user_subgraphs(
                u,
                data,
                graph,
                item_max_length,
                user_max_length,
                k_hop,
            )
        )(u)
        for u in user
    )
    return sample_lists


def _correct_shape(sample_lists: List) -> List:
    sample_list = [item for sample_list in sample_lists for item in sample_list]
    return sample_list


def preprocess_dgsr(
    df: pd.DataFrame,
    graph: dgl.DGLGraph,
    item_max_length: int = 50,
    user_max_length: int = 50,
    job: int = 10,
    k_hop: int = 2,
):
    """Preprocess raw transaction data and graph created from all transaction into user's transactions subgraphs which
    are final model inputs.

    Args:
        data (pd.DataFrame): dataframe with mapped transactions
        graph (dgl.DGLGraph): graph object created from whole dataset
        item_max_length (int): maximum length of item interactions for single user
        user_max_length (int): maximum number of users to sample into subgraph
        job (int): number of maximum concurrently running jobs for Parallel function from joblib module
        k_hop (int): number of iterations for subgraphs creation

    Returns:
        Tuple: (train_list, val_list, test_list) - each one contains subgraphs for train/val/test subsets
    """
    df = _concat_chunks(df)
    df = _simple_preprocess(df)
    sample_lists = _generate_model_input(
        df,
        graph,
        item_max_length,
        user_max_length,
        job=job,
        k_hop=k_hop,
    )
    train_list, val_list, test_list = ([], [], [])
    for train, val, test in sample_lists:
        train_list.append(train)
        val_list.append(val)
        test_list.append(test)
    train_list = _correct_shape(train_list)
    val_list = _correct_shape(val_list)
    test_list = _correct_shape(test_list)
    logger.info(f"The number of samples in train set: {len(train_list)}")
    logger.info(f"The number of samples in val set: {len(val_list)}")
    logger.info(f"The number of samples in test set: {len(test_list)}")
    return train_list, val_list, test_list


def negative_sample_dgsr(df: pd.DataFrame) -> pd.DataFrame:
    """Sample negative items for each user based on transactions dataframe

    Args:
        data (pd.DataFrame): dataframe with mapped transactions

    Returns:
        pd.DataFrame: dataframe with sample of negative items for each user
    """
    df = _concat_chunks(df)
    df = df.drop("time", axis=1)
    item = df.loc[:, "item_id"].unique()
    item_num = len(item)
    data_neg = user_neg(df, item_num)
    return data_neg
