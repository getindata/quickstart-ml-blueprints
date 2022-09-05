import logging
from typing import List

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


def _cal_order(data):
    data = data.sort_values(["time"], kind="mergesort")
    data["order"] = range(len(data))
    return data


def _cal_u_order(data):
    data = data.sort_values(["time"], kind="mergesort")
    data["u_order"] = range(len(data))
    return data


def _refine_time(data):
    data = data.sort_values(["time"], kind="mergesort")
    time_seq = data["time"].values
    time_gap = 1
    for i, da in enumerate(time_seq[0:-1]):
        if time_seq[i] == time_seq[i + 1] or time_seq[i] > time_seq[i + 1]:
            time_seq[i + 1] = time_seq[i + 1] + time_gap
            time_gap += 1
    data["time"] = time_seq
    return data


def _simple_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby("user_id").apply(_refine_time).reset_index(drop=True)
    df["time"] = df["time"].astype("int64")
    return df


def generate_graph_dgsr(df: pd.DataFrame) -> dgl.DGLGraph:
    df = _concat_chunks(df)
    data = _simple_preprocess(df)
    data = data.groupby("user_id").apply(_refine_time).reset_index(drop=True)
    data = data.groupby("user_id").apply(_cal_order).reset_index(drop=True)
    data = data.groupby("item_id").apply(_cal_u_order).reset_index(drop=True)
    user = data["user_id"].values
    item = data["item_id"].values
    time = data["time"].values
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


def _generate_user(
    user,
    data,
    graph,
    item_max_length,
    user_max_length,
    k_hop=3,
):
    data_user = data[data["user_id"] == user].sort_values("time")
    u_time = data_user["time"].values
    u_seq = data_user["item_id"].values
    split_point = len(u_seq) - 1
    train_list = []
    val_list = []
    test_list = []
    if len(u_seq) < 3:
        return train_list, val_list, test_list
    else:
        for j, t in enumerate(u_time[0:-1]):
            if j == 0:
                continue
            if j < item_max_length:
                start_t = u_time[0]
            else:
                start_t = u_time[j - item_max_length]
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
            his_user = torch.tensor([user])
            graph_i = select_topk(
                sub_graph, item_max_length, weight="time", nodes={"user": u_temp}
            )
            i_temp = torch.unique(graph_i.edges(etype="by")[0])
            his_item = torch.unique(graph_i.edges(etype="by")[0])
            edge_i = [graph_i.edges["by"].data[dgl.NID]]
            edge_u = []
            for _ in range(k_hop - 1):
                graph_u = select_topk(
                    sub_graph, user_max_length, weight="time", nodes={"item": i_temp}
                )
                u_temp = np.setdiff1d(
                    torch.unique(graph_u.edges(etype="pby")[0]), his_user
                )[-user_max_length:]
                graph_i = select_topk(
                    sub_graph, item_max_length, weight="time", nodes={"user": u_temp}
                )
                his_user = torch.unique(torch.cat([torch.tensor(u_temp), his_user]))
                i_temp = np.setdiff1d(
                    torch.unique(graph_i.edges(etype="by")[0]), his_item
                )
                his_item = torch.unique(torch.cat([torch.tensor(i_temp), his_item]))
                edge_i.append(graph_i.edges["by"].data[dgl.NID])
                edge_u.append(graph_u.edges["pby"].data[dgl.NID])
            all_edge_u = torch.unique(torch.cat(edge_u))
            all_edge_i = torch.unique(torch.cat(edge_i))
            fin_graph = dgl.edge_subgraph(
                sub_graph, edges={"by": all_edge_i, "pby": all_edge_u}
            )
            target = u_seq[j + 1]
            last_item = u_seq[j]
            u_alis = torch.where(fin_graph.nodes["user"].data["user_id"] == user)[0]
            last_alis = torch.where(
                fin_graph.nodes["item"].data["item_id"] == last_item
            )[0]
            graph_dict = {
                "user": torch.tensor([user]),
                "target": torch.tensor([target]),
                "u_alis": u_alis,
                "last_alis": last_alis,
            }
            if j < split_point - 1:
                train_list.append([user, j, fin_graph, graph_dict])
            if j == split_point - 1 - 1:
                val_list.append([user, j, fin_graph, graph_dict])
            if j == split_point - 1:
                test_list.append([user, j, fin_graph, graph_dict])
    return train_list, val_list, test_list


def _generate_data(
    data,
    graph,
    item_max_length,
    user_max_length,
    job=10,
    k_hop=3,
):
    user = data["user_id"].unique()
    sample_lists = Parallel(n_jobs=job)(
        delayed(
            lambda u: _generate_user(
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
    df = _concat_chunks(df)
    df = _simple_preprocess(df)
    sample_lists = _generate_data(
        df,
        graph,
        item_max_length,
        user_max_length,
        job=job,
        k_hop=k_hop,
    )
    train_list = []
    val_list = []
    test_list = []
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
    df = _concat_chunks(df)
    item = df.loc[:, "item_id"]
    item_num = len(item)
    data_neg = user_neg(df, item_num)
    return data_neg
