import logging
import os

import dgl
import numpy as np
import pandas as pd
import torch
from dgl import save_graphs
from dgl.sampling import select_topk
from joblib import Parallel, delayed

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


def generate_graph(data):
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


def generate_user(
    user,
    data,
    graph,
    item_max_length,
    user_max_length,
    train_path,
    test_path,
    k_hop=3,
    val_path=None,
):
    data_user = data[data["user_id"] == user].sort_values("time")
    u_time = data_user["time"].values
    u_seq = data_user["item_id"].values
    split_point = len(u_seq) - 1
    train_num = 0
    test_num = 0
    if len(u_seq) < 3:
        return 0, 0
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
            if j < split_point - 1:
                save_graphs(
                    train_path
                    + "/"
                    + str(user)
                    + "/"
                    + str(user)
                    + "_"
                    + str(j)
                    + ".bin",
                    fin_graph,
                    {
                        "user": torch.tensor([user]),
                        "target": torch.tensor([target]),
                        "u_alis": u_alis,
                        "last_alis": last_alis,
                    },
                )
                train_num += 1
            if j == split_point - 1 - 1:
                save_graphs(
                    val_path
                    + "/"
                    + str(user)
                    + "/"
                    + str(user)
                    + "_"
                    + str(j)
                    + ".bin",
                    fin_graph,
                    {
                        "user": torch.tensor([user]),
                        "target": torch.tensor([target]),
                        "u_alis": u_alis,
                        "last_alis": last_alis,
                    },
                )
            if j == split_point - 1:
                save_graphs(
                    test_path
                    + "/"
                    + str(user)
                    + "/"
                    + str(user)
                    + "_"
                    + str(j)
                    + ".bin",
                    fin_graph,
                    {
                        "user": torch.tensor([user]),
                        "target": torch.tensor([target]),
                        "u_alis": u_alis,
                        "last_alis": last_alis,
                    },
                )
                test_num += 1
        return train_num, test_num


def _generate_data(
    data,
    graph,
    item_max_length,
    user_max_length,
    train_path,
    test_path,
    val_path,
    job=10,
    k_hop=3,
):
    user = data["user_id"].unique()
    a = Parallel(n_jobs=job)(
        delayed(
            lambda u: generate_user(
                u,
                data,
                graph,
                item_max_length,
                user_max_length,
                train_path,
                test_path,
                k_hop,
                val_path,
            )
        )(u)
        for u in user
    )
    return a


def _simple_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby("user_id").apply(_refine_time).reset_index(drop=True)
    df["time"] = df["time"].astype("int64")
    return df


def generate_graph_dgsr(df: pd.DataFrame) -> dgl.DGLGraph:
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


def preprocess_dgsr(
    df: pd.DataFrame,
    graph: dgl.DGLGraph,
    item_max_length: int = 50,
    user_max_length: int = 50,
    job: int = 10,
    k_hop: int = 2,
):
    df = _simple_preprocess(df)
    path = "data/04_feature/santander/"
    train_path = os.path.join(path, "/train/")
    val_path = os.path.join(path, "/val/")
    test_path = os.path.join(path, "/test/")
    all_num = _generate_data(
        df,
        graph,
        item_max_length,
        user_max_length,
        train_path,
        test_path,
        val_path,
        job=job,
        k_hop=k_hop,
    )
    train_num = 0
    test_num = 0
    for num_ in all_num:
        train_num += num_[0]
        test_num += num_[1]
    logger.info(f"The number of samples in train set: {train_num}")
    logger.info(f"The number of samples in test set: {test_num}")


def negative_sample_dgsr(df: pd.DataFrame) -> pd.DataFrame:
    item = df.loc[:, "item_id"]
    item_num = len(item)
    data_neg = user_neg(df, item_num)
    return data_neg
