"""Some variables will have names form the original paper script for easier comparison and understanding"""
import logging
from typing import Dict, List, Tuple
from xmlrpc.client import Boolean

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.sampling import select_topk
from joblib import Parallel, delayed

from recommender_gnn.extras.datasets.chunks_dataset import _concat_chunks
from recommender_gnn.extras.graph_utils.dgsr_utils import user_neg

pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)


def _sort_transactions(data: pd.DataFrame, colname: str) -> pd.DataFrame:
    data = data.sort_values(["time"], kind="mergesort")
    data.loc[:, colname] = range(len(data))
    return data


def _refine_time(data: pd.DataFrame) -> pd.DataFrame:
    """Adds time gaps to events with same timestamp to introduce some kind of order"""
    data = data.sort_values(["time"], kind="mergesort")
    time_seq = data.loc[:, "time"].values
    time_gap = 1
    for i, _ in enumerate(time_seq[0:-1]):
        if time_seq[i] == time_seq[i + 1] or time_seq[i] > time_seq[i + 1]:
            time_seq[i + 1] = time_seq[i + 1] + time_gap
            time_gap += 1
    data.loc[:, "time"] = time_seq
    return data


def _preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby("user_id").apply(_refine_time).reset_index(drop=True)
    df.loc[:, "time"] = df.loc[:, "time"].astype("int64")
    return df


def _construct_graph(user: List, item: List, time: List) -> dgl.DGLGraph:
    """Constructs heterogeneous bidirectional graph object with interactions
    between users and items with information about timestamps.
    """
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


def generate_graph_dgsr(df: pd.DataFrame) -> dgl.DGLGraph:
    """Generates graph from whole dataset. This graph will be used for creating transactions subgraphs for each user.

    Args:
        df (pd.DataFrame): dataframe with mapped transactions

    Returns:
        dgl.DGLGraph: graph object created from transaction in input dataframe
    """
    df = _concat_chunks(df)
    user_column, item_column, time_column = ("user_id", "item_id", "time")
    data = _preprocess_transactions(df)
    data = data.groupby(user_column).apply(_refine_time).reset_index(drop=True)
    data = (
        data.groupby(user_column)
        .apply(lambda x: _sort_transactions(x, "order"))
        .reset_index(drop=True)
    )
    data = (
        data.groupby(item_column)
        .apply(lambda x: _sort_transactions(x, "u_order"))
        .reset_index(drop=True)
    )
    user = data.loc[:, user_column].values
    item = data.loc[:, item_column].values
    time = data.loc[:, time_column].values
    graph = _construct_graph(user, item, time)
    return graph


def _conditional_compare(
    timestamp_1: np.array,
    timestamp_2: np.array,
    condition: Boolean,
) -> Tuple[Boolean]:
    return timestamp_1 <= timestamp_2 if condition else timestamp_1 < timestamp_2


def _get_sub_edges(
    graph: dgl.DGLGraph,
    u_time: List,
    predict_condition: bool,
    start_t: int,
    next_index: int,
) -> Tuple[List]:
    """Return boolean vectors indicating subsets of edges between users and items based on specified timeframe.

    Args:
        graph (dgl.DGLGraph): graph object created from whole dataset
        u_time (List): list of all timestamps for a user in a graph
        predict_condition (bool): boolean indicating if we are in prediction mode, where we are building subgraphs
            for a user based on all previous interactions, not excluding the last one for verification purposes
        start_t (int): starting timestamp for a subgraph
        next_index (int): index of the next timestamp for a given subgraph

    Returns:
        Tuple: (sub_u_eid, sub_i_eid) - boolean vectors indicating subsets of edges between users and items for a given
            timeframe
    """
    sub_u_eid = (
        _conditional_compare(
            graph.edges["by"].data["time"], u_time[next_index], predict_condition
        )
    ) & (graph.edges["by"].data["time"] >= start_t)
    sub_i_eid = (
        _conditional_compare(
            graph.edges["pby"].data["time"], u_time[next_index], predict_condition
        )
    ) & (graph.edges["pby"].data["time"] >= start_t)
    return (sub_u_eid, sub_i_eid)


def _generate_subgraphs(
    graph: dgl.DGLGraph,
    u_time: List,
    start_t: int,
    user: int,
    item_max_length: int,
    seq_item: int,
    seq_len: int,
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.tensor]:
    """Generates subgraphs for chosen user based on selected transactions timestamps.

    Args:
        graph (dgl.DGLGraph): graph object created from whole dataset
        u_time (List): list of all timestamps for a user in a graph
        start_t (int): starting timestamp for a subgraph
        user (int): user id
        item_max_length (int): maximum number of items in a generated subgraph
        seq_item (int): id of last item in a user interactions sequence considered for building a subgraph, parameter
            that defines and limits the size of a subgraph
        seq_len (int): length of a all user interactions sequence

    """
    next_index = seq_item
    predict_condition = seq_item == (seq_len - 1)
    if not predict_condition:
        next_index += 1
    sub_u_eid, sub_i_eid = _get_sub_edges(
        graph, u_time, predict_condition, start_t, next_index
    )
    sub_graph = dgl.edge_subgraph(
        graph, edges={"by": sub_u_eid, "pby": sub_i_eid}, relabel_nodes=False
    )
    current_user = torch.tensor([user])
    graph_i = select_topk(
        sub_graph, item_max_length, weight="time", nodes={"user": current_user}
    )
    return graph_i, sub_graph, current_user


def _iterate_k_hops(
    k_hop: int,
    sub_graph: dgl.DGLGraph,
    user_max_length: int,
    item_max_length: int,
    user: int,
    current_items: int,
    graph_i: dgl.DGLGraph,
) -> Tuple[List]:
    """Generates final subset of edges for a user subgraph by sampling user and item edges from k_hop iteration through
    previously created subgraph. Thanks to this approach we are limiting the size of a final subgraph and choosing only
    the most relevant interactions.

    Args:
        k_hop (int): number of iterations through a subgraph
        sub_graph (dgl.DGLGraph): full intermediate subgraph for given user
        user_max_length (int): maximum number of users in a generated final subgraph
        item_max_length (int): maximum number of items in a generated final subgraph
        user (int): user id
        current_items (int): list of items in a subgraph
        graph_i (dgl.DGLGraph): subgraph of users after applying initial select_topk function

    Returns:
        Tuple: (edge_i, edge_u) - edges for final subgraph after sampling

    """
    edge_i = [graph_i.edges["by"].data[dgl.NID]]
    edge_u = []
    temp_users = user
    temp_items = current_items
    for _ in range(k_hop - 1):
        graph_u = select_topk(
            sub_graph, user_max_length, weight="time", nodes={"item": current_items}
        )
        current_users = np.setdiff1d(
            torch.unique(graph_u.edges(etype="pby")[0]), temp_users
        )[-user_max_length:]
        graph_i = select_topk(
            sub_graph, item_max_length, weight="time", nodes={"user": current_users}
        )
        temp_users = torch.unique(torch.cat([torch.tensor(current_users), temp_users]))
        current_items = np.setdiff1d(
            torch.unique(graph_i.edges(etype="by")[0]), temp_items
        )
        temp_items = torch.unique(torch.cat([torch.tensor(current_items), temp_items]))
        edge_i.append(graph_i.edges["by"].data[dgl.NID])
        edge_u.append(graph_u.edges["pby"].data[dgl.NID])
    return (edge_i, edge_u)


def _sample_final_subgraph(
    sub_graph: dgl.DGLGraph,
    graph_i: dgl.DGLGraph,
    k_hop: int,
    user_max_length: int,
    item_max_length: int,
    user: int,
) -> Tuple:
    """Iterates through subgraph to sample neighbors and creates final subgraph for a given user.

    Args:
        sub_graph (dgl.DGLGraph): full intermediate subgraph for given user
        graph_i (dgl.DGLGraph): subgraph of users after applying initial select_topk function
        k_hop (int): number of iterations through a subgraph
        user_max_length (int): maximum number of users in a generated final subgraph
        item_max_length (int): maximum number of items in a generated final subgraph
        user (int): user id

    """
    current_item = torch.unique(graph_i.edges(etype="by")[0])
    edge_i, edge_u = _iterate_k_hops(
        k_hop,
        sub_graph,
        user_max_length,
        item_max_length,
        user,
        current_item,
        graph_i,
    )
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
    seq_item: int,
    seq_len: int,
) -> Dict:
    """Prepare dictionary object with graph data.

    Args:
        fin_graph (dgl.DGLGraph): final subgraph for given user
        user (int): user id
        u_seq (List): list of all items for a user in a graph
        seq_item (int): id of last item in a user interactions sequence considered for building a subgraph, parameter
            that defines and limits the size of a subgraph
        seq_len (int): length of a all user interactions sequence
    """
    # No target/label for predicting unseen data
    if seq_item == (seq_len - 1):
        target = np.int64(0)
    else:
        target = u_seq[seq_item + 1]
    last_item = u_seq[seq_item]
    last_user_alias = torch.where(fin_graph.nodes["user"].data["user_id"] == user)[0]
    last_item_alias = torch.where(fin_graph.nodes["item"].data["item_id"] == last_item)[
        0
    ]
    graph_dict = {
        "user": torch.tensor([user]),
        "target": torch.tensor([target]),
        "u_alias": last_user_alias,
        "last_alias": last_item_alias,
    }
    return graph_dict


def _prepare_user_data(data: pd.DataFrame, user: int) -> Tuple:
    data_user = data[data["user_id"] == user].sort_values("time")
    data_user["time"] = data_user["time"].astype("int32")
    u_time = data_user["time"].values
    u_seq = data_user["item_id"].values
    return u_seq, u_time


def _split_subgraphs(all_subgraphs: List, subsets: List[Boolean]) -> Tuple[List]:
    """Train/val/test split based on order in a transactions sequence. We can generate train, train + val,
    train + val + test or train + predict subsets. For example, for selected user with 6
    transactions in case of (train, val, test, predict) we will have: subgraphs of size 6 for predict subset,
    subgraph of size 5 for test subset, subgraph of size 4 for val subset and subgraphs of size 3 and 2 for train
    subset.
    """
    _, _, predict_flag = subsets
    n_subsets = sum(subsets[0:2]) + 1
    train_list, val_list, test_list, predict_list = ([], [], [], [])
    if predict_flag:
        predict_list, all_subgraphs = all_subgraphs[-1:], all_subgraphs[:-1]
    if n_subsets == 3:
        test_list, all_subgraphs = all_subgraphs[-1:], all_subgraphs[:-1]
    # Strange train/val split but leaving as it was in original implementation
    if n_subsets > 1 and all_subgraphs[:-1]:
        val_list, train_list = all_subgraphs[-1:], all_subgraphs[:-1]
    else:
        train_list = all_subgraphs
    return train_list, val_list, test_list, predict_list


def _generate_user_subgraphs(
    user: int,
    data: pd.DataFrame,
    graph: dgl.DGLGraph,
    item_max_length: int,
    user_max_length: int,
    k_hop: int,
    subsets: List[Boolean],
) -> Tuple[List]:
    """Generates subgraphs of transactions for each users interaction. For example if user has 6 transactions it will
    generate subgraphs of size 6,5,4,3 and 2 for more training examples and split them into train/val/test/predict
    subsets.

    Args:
        user (int): user id from mapped transactions dataframe
        data (pd.DataFrame): dataframe with mapped transactions
        graph (dgl.DGLGraph): graph object created from whole dataset
        item_max_length (int): maximum length of item interactions for single user
        user_max_length (int): maximum number of users to sample into subgraph
        k_hop (int): number of iterations for subgraphs creation
        subsets (List[Boolean]): list of booleans which indicates if we want to generate subgraphs for val, test,
            predict subsets accordingly. Always generating subset for train set. Considered possibilities are train,
            train + val, train + val + test and train + predict subsets. For predicting we are using full sized
            subgraphs containing all transactions, because we want to use all information and because we do not need
            true labels for verification.

    Returns:
        Tuple: (train_list, val_list, test_list, predict_list) - each one contains subgraphs for train/val/test/predict
        subsets
    """
    predict_flag = subsets[2]
    u_seq, u_time = _prepare_user_data(data, user)
    train_list, val_list, test_list, predict_list, all_subgraphs = ([], [], [], [], [])
    u_time_seq = u_time[0:-1]
    if predict_flag:
        u_time_seq = u_time
    # Considering only users with at least two transactions
    seq_len = len(u_time_seq)
    if seq_len < 2:
        return train_list, val_list, test_list, predict_list
    else:
        for seq_item, _ in enumerate(u_time_seq):
            if seq_item == 0:
                continue
            if seq_item < item_max_length:
                start_t = u_time[0]
            else:
                start_t = u_time[seq_item - item_max_length]
            graph_i, sub_graph, current_user = _generate_subgraphs(
                graph, u_time, start_t, user, item_max_length, seq_item, seq_len
            )
            fin_graph = _sample_final_subgraph(
                sub_graph,
                graph_i,
                k_hop,
                user_max_length,
                item_max_length,
                current_user,
            )
            graph_dict = _prepare_graph_dict(fin_graph, user, u_seq, seq_item, seq_len)
            all_subgraphs.append([user, seq_item, fin_graph, graph_dict])
        train_list, val_list, test_list, predict_list = _split_subgraphs(
            all_subgraphs, subsets
        )
    return train_list, val_list, test_list, predict_list


def _generate_model_input(
    data: pd.DataFrame,
    graph: dgl.DGLGraph,
    item_max_length: int,
    user_max_length: int,
    k_hop: int,
    subsets: List,
):
    """Wrapper for generating subgraphs for each user from transactions dataframe."""
    user = data.loc[:, "user_id"].unique()
    user_subset_lists = Parallel(n_jobs=10)(
        delayed(
            lambda u: _generate_user_subgraphs(
                u,
                data,
                graph,
                item_max_length,
                user_max_length,
                k_hop,
                subsets,
            )
        )(u)
        for u in user
    )
    return user_subset_lists


def _correct_shape(sample_lists: List) -> List:
    sample_list = [item for sample_list in sample_lists for item in sample_list]
    return sample_list


def preprocess_dgsr(
    df: pd.DataFrame,
    graph: dgl.DGLGraph,
    item_max_length: int = 50,
    user_max_length: int = 50,
    k_hop: int = 3,
    val_flag: Boolean = False,
    test_flag: Boolean = True,
    predict_flag: Boolean = False,
):
    """Preprocess raw transaction data and graph created from all transaction into user's transactions subgraphs which
    are final model inputs. Considered possibilities for generated subsets are train, train + val,
    train + val + test and train + predict subsets.

    Args:
        data (pd.DataFrame): dataframe with mapped transactions
        graph (dgl.DGLGraph): graph object created from whole dataset
        item_max_length (int): maximum length of item interactions for single user
        user_max_length (int): maximum number of users to sample into subgraph
        k_hop (int): number of iterations for subgraphs creation
        val_flag (Boolean): if we want to generate validation subset
        test_flag (Boolean): if we want to generate test subset
        predict_flag (Booleand): if we want to generate predict subset For predicting we are using full sized
            subgraphs containing all transactions, because we want to use all information and because we do not need
            true labels for verification.

    Returns:
        Tuple: (train_list, val_list, test_list, predict_list) - each one contains subgraphs for train/val/test/predict
        subsets
    """
    subsets = [val_flag, test_flag, predict_flag]
    df = _concat_chunks(df)
    df = _preprocess_transactions(df)
    sample_lists = _generate_model_input(
        df,
        graph,
        item_max_length,
        user_max_length,
        k_hop=k_hop,
        subsets=subsets,
    )
    train_list, val_list, test_list, predict_list = ([], [], [], [])
    for train, val, test, predict in sample_lists:
        train_list.append(train)
        val_list.append(val)
        test_list.append(test)
        predict_list.append(predict)
    train_list = _correct_shape(train_list)
    val_list = _correct_shape(val_list)
    test_list = _correct_shape(test_list)
    predict_list = _correct_shape(predict_list)
    logger.info(f"The number of samples in train set: {len(train_list)}")
    logger.info(f"The number of samples in val set: {len(val_list)}")
    logger.info(f"The number of samples in test set: {len(test_list)}")
    logger.info(f"The number of samples in predict set: {len(predict_list)}")
    return train_list, val_list, test_list, predict_list


def sample_negatives_dgsr(df: pd.DataFrame) -> pd.DataFrame:
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
