import logging
from typing import Dict, Iterable, List, Set, Union

import pandas as pd
from sklearn.neighbors import KDTree

from gid_ml_framework.helpers.utils import filter_dataframe_by_last_n_days

logger = logging.getLogger(__name__)
ArticlesSet = Set
Bin = str
ArticlesBin = Dict[Bin, ArticlesSet]


# GLOBAL (for all users) ARTICLES
def _most_sold_articles(transactions: pd.DataFrame, top_k: int) -> ArticlesSet:
    """Calculates `top_k` most sold articles by count for given dataframe.

    Args:
        transactions (pd.DataFrame): transactions
        top_k (int): number of most sold articles to keep

    Returns:
        ArticlesSet: set of `article_id`
    """
    articles_sold_most = (
        transactions.groupby(["article_id"])["customer_id"]
        .size()
        .reset_index(name="sales_sum_count")
        .sort_values(by="sales_sum_count", ascending=False)
        .head(top_k)["article_id"]
    )
    articles_set = set(articles_sold_most)
    return articles_set


def collect_global_articles(
    transactions: pd.DataFrame,
    n_days_list: List[Union[int, None]],
    top_k_list: List[int],
) -> ArticlesSet:
    """Collects a set of most sold articles from different `n_days`.

    Args:
        transactions (pd.DataFrame): transactions
        n_days_list (List[Union[int, None]]):
        list of different time periods (latest `n_days`) for which to calculate most sold items
        top_k_list (List[int]): for each time period, how many most sold items to select (`top_k`)

    Returns:
        ArticlesSet: distinct set of articles for all time periods (`n_days_list`) for most sold items (`top_k_list`)
    """
    all_global_articles = set()
    for n_days, top_k in zip(n_days_list, top_k_list):
        latest_transactions = filter_dataframe_by_last_n_days(
            transactions, n_days, date_column="t_dat"
        )
        articles_set = _most_sold_articles(latest_transactions, top_k)
        logger.info(
            f"All articles size: {len(all_global_articles)} before adding articles from {n_days=}, {top_k=}"
        )
        all_global_articles.update(articles_set)
        logger.info(f"All articles size after: {len(all_global_articles)}")
    return all_global_articles


def assign_global_articles(
    customers: pd.DataFrame, articles: ArticlesSet
) -> pd.DataFrame:
    """Creates a dataframe with global articles (most sold items) for each customer.

    Args:
        customers (pd.DataFrame): customers
        articles (ArticlesSet): set of most sold articles

    Returns:
        pd.DataFrame: dataframe with: customer_id and global_articles (list of `article_id`)
    """
    all_customers = customers[["customer_id"]].copy()
    logger.info(f"Number of all customers: {len(all_customers)}")
    all_customers.loc[:, "global_articles"] = [
        list(articles) for i in all_customers.index
    ]
    logger.info(f"{all_customers.shape=}")
    return all_customers


# SEGMENT ARTICLES
def segment_by_customer_age(customers: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Creates customer segments based on age.

    Args:
        customers (pd.DataFrame): customers
        n_bins (int): number of segments

    Returns:
        pd.DataFrame: dataframe with: customer_id and segment_bin (age segment)
    """
    customers["segment_bin"] = (
        pd.qcut(customers["age"], q=n_bins, labels=False).astype(str).str.zfill(2)
    )
    return customers[["customer_id", "segment_bin"]]


def _filter_transactions_by_bin(
    transactions: pd.DataFrame, customers: pd.DataFrame, bin: str
) -> pd.DataFrame:
    """Keeps transactions for given age segment.

    Args:
        transactions (pd.DataFrame): transactions
        customers (pd.DataFrame): customers with age segment
        bin (str): age segment for filtering

    Returns:
        pd.DataFrame: filtered transactions dataframe
    """
    filtered_customers = customers[customers["segment_bin"] == bin]
    filtered_transactions = transactions.merge(
        filtered_customers[["customer_id"]], on="customer_id"
    )
    return filtered_transactions


def _most_sold_articles_by_segment(
    transactions: pd.DataFrame, customers_bins: pd.DataFrame, top_k: int
) -> ArticlesBin:
    """Calculates most sold articles for each age segment.

    Args:
        transactions (pd.DataFrame): transactions
        customers_bins (pd.DataFrame): customers with age segment
        top_k (int): number of most sold items to keep

    Returns:
        ArticlesBin: dictionary with age segment as key, and set of most sold articles for that age segment as values
    """
    articles_per_bin_dict = dict()
    unique_bins = list(customers_bins["segment_bin"].unique())
    for bin in unique_bins:
        bin_transactions = _filter_transactions_by_bin(
            transactions, customers_bins, bin
        )
        articles_set = _most_sold_articles(bin_transactions, top_k)
        articles_per_bin_dict[bin] = articles_set
    return articles_per_bin_dict


def _update_dict_of_sets(cumulative_dict: Dict, new_dict: Dict) -> Dict:
    """Appends `cumulative_dict` with items from `new_dict`.

    Args:
        cumulative_dict (Dict): dictionary to be appended
        new_dict (Dict): dictionary with new values for `cumulative_dict`

    Returns:
        Dict: dictionary with all values
    """
    for k, v in new_dict.items():
        cumulative_dict[k] = cumulative_dict.get(k, set()) | v
        logger.info(
            f"Bin: {k=}, size of set: {len(v)=}, size of acc. set: {len(cumulative_dict[k])=}"
        )
    return cumulative_dict


def collect_segment_articles(
    transactions: pd.DataFrame,
    customers_bins: pd.DataFrame,
    n_days_list: List[Union[int, None]],
    top_k_list: List[int],
) -> ArticlesBin:
    """Collects a dictionary with most sold items for each age segment from different `n_days`.

    Args:
        transactions (pd.DataFrame): transactions
        customers_bins (pd.DataFrame): customers with age segment
        n_days_list (List[Union[int, None]]):
        list of different time periods (latest `n_days`) for which to calculate most sold items
        top_k_list (List[int]): for each time period, how many most sold items to select (`top_k`)

    Returns:
        ArticlesBin: dictionary with age segment as key,
            and set of most sold articles for that age segment as values for different time periods (`n_days_list`)
    """
    segment_articles = dict()
    for n_days, top_k in zip(n_days_list, top_k_list):
        latest_transactions = filter_dataframe_by_last_n_days(
            transactions, n_days, date_column="t_dat"
        )
        articles_per_bin_dict = _most_sold_articles_by_segment(
            latest_transactions, customers_bins, top_k
        )
        segment_articles = _update_dict_of_sets(segment_articles, articles_per_bin_dict)
    return segment_articles


def assign_segment_articles(
    articles_dict: ArticlesBin, customers_segment: pd.DataFrame
) -> pd.DataFrame:
    """Creates a dataframe with segment articles (most sold items by age segment) for each customer.

    Args:
        articles_dict (ArticlesBin): dictionary with age segment as key, and set of most sold articles
        customers_segment (pd.DataFrame): customers with age segment

    Returns:
        pd.DataFrame: dataframe with: customer_id and segment_articles (list of `article_id`)
    """
    articles_df = pd.DataFrame(articles_dict.keys(), columns=["segment_bin"])
    articles_df["segment_articles"] = articles_df.loc[:, "segment_bin"].map(
        articles_dict
    )
    segment_articles = customers_segment.merge(articles_df, on="segment_bin").drop(
        ["segment_bin"], axis=1
    )
    return segment_articles


# PREVIOUSLY BOUGHT ARTICLES
def collect_previously_bought_articles(transactions: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe with list of previously bought `article_id` for each customer.

    Args:
        transactions (pd.DataFrame): transactions

    Returns:
        pd.DataFrame: dataframe with: customer_id and previously_bought (list of `article_id`)
    """
    prev_bought = transactions[["customer_id", "article_id"]].drop_duplicates()
    prev_bought = (
        prev_bought.groupby(["customer_id"])["article_id"]
        .apply(list)
        .reset_index(name="previously_bought")
    )
    return prev_bought


def collect_previously_bought_prod_name_articles(
    transactions: pd.DataFrame, articles: pd.DataFrame
) -> pd.DataFrame:
    """Returns a dataframe with list of `article_id` from the same `prod_name`
    for which customer previously bought article from.
    For example: if customer A bought `prod_name='Socks'` then,
    the resulting dataframe will contain all items from `prod_name='Socks'`.

    Args:
        transactions (pd.DataFrame): transactions
        articles (pd.DataFrame): articles

    Returns:
        pd.DataFrame: dataframe with: customer_id and previously_bought_prod_name (list of `article_id`)
    """
    articles_prod_name = articles[["article_id", "prod_name"]]
    customers_prod_name = transactions.merge(articles_prod_name, on="article_id")[
        ["customer_id", "prod_name"]
    ].drop_duplicates()
    prev_bought_prod_name = customers_prod_name.merge(
        articles_prod_name, on="prod_name"
    ).drop(["prod_name"], axis=1)
    prev_bought_prod_name = (
        prev_bought_prod_name.groupby(["customer_id"])["article_id"]
        .apply(list)
        .reset_index(name="previously_bought_prod_name")
    )
    return prev_bought_prod_name


# SIMILAR IMAGES/TEXT EMBEDDINGS
def _build_tree(embeddings: pd.DataFrame) -> KDTree:
    """Builds a KDTree object

    Args:
        embeddings (pd.DataFrame): embeddings

    Returns:
        KDTree: k-dimensional tree
    """
    logger.info("Building KDTree")
    tree = KDTree(embeddings.values, leaf_size=5)
    return tree


def _find_similar_vectors(
    query_id: str, embeddings: pd.DataFrame, tree: KDTree, k_closest: int
) -> Union[List[str], None]:
    """Returns indices of `k_closest` vectors for `query_id` item inside `embeddings` dataframe.
    If not found (for instance -> missing picture), returns `None`.

    Args:
        query_id (str): item for which to find closest vectors
        embeddings (pd.DataFrame): dataframe with all embedding vectors
        tree (KDTree): built (trained) KDTree
        k_closest (int): number of closest vectors (indices) to return

    Returns:
        Union[List[str], None]: list of closest indices, or `None` if not found in embeddings dataframe
    """
    try:
        _, ind = tree.query(embeddings.loc[query_id].values.reshape(1, -1), k=k_closest)
    except KeyError:
        return None
    closest_embedding_idx = embeddings.iloc[ind[0]].index.tolist()
    return closest_embedding_idx


def _create_embedding_dictionary(
    items: Iterable[str], embeddings: pd.DataFrame, k_closest: int
) -> Dict:
    """Creates embedding dictionary of `k_closest` indices for each item.

    Args:
        items (Iterable[str]): all items to be queried
        embeddings (pd.DataFrame): dataframe with all embedding vectors
        k_closest (int): number of closest vectors (indices) to return

    Returns:
        Dict: embedding dictionary of `k_closest` indices for each item
    """
    tree = _build_tree(embeddings)
    closest_dict = dict()
    logger.info("Started querying similar vectors")
    for item in items:
        if similar_vectors := _find_similar_vectors(item, embeddings, tree, k_closest):
            closest_dict[item] = similar_vectors
    logger.info("Finished querying similar vectors")
    return closest_dict


def _cleanup_closest_embeddings(embeddings: pd.DataFrame, name: str) -> pd.DataFrame:
    """Removes NAs, duplicates (multiple items, same closest vectors),
    and returns a dataframe with customer_id and distinct list of `article_id`.

    Args:
        embeddings (pd.DataFrame): dataframe with all embedding vectors
        name (str): column name for closest embeddings indices

    Returns:
        pd.DataFrame: dataframe with: customer_id and name (list of distinct `article_id`)
    """
    logger.info(f"Closest embeddings df shape before cleanup: {embeddings.shape}")
    embeddings.dropna(axis=0, how="any", inplace=True)
    embeddings = embeddings.explode(column=name)
    embeddings.drop_duplicates(inplace=True)
    # filter out querying item from results
    embeddings = embeddings[embeddings[name] != embeddings["article_id"]]
    embeddings.drop(["article_id"], axis=1, inplace=True)
    embeddings = (
        embeddings.groupby(["customer_id"])[name].apply(list).reset_index(name=name)
    )
    logger.info(f"Closest embeddings df shape after cleanup: {embeddings.shape}")
    return embeddings


def collect_similar_embeddings(
    transactions: pd.DataFrame,
    embeddings: pd.DataFrame,
    n_last_bought: int = 5,
    k_closest: int = 5,
    name: str = "closest_emb",
) -> pd.DataFrame:
    """Returns a dataframe with `k_closest` embeddings indices for each of `n_last_bought` items for each customer.
    Transactions are sorted by date descendingly.

    Args:
        transactions (pd.DataFrame): transactions
        embeddings (pd.DataFrame): dataframe with all embedding vectors
        n_last_bought (int, optional): max number of last items for which to look for similar items for each customer.
            Defaults to 5.
        k_closest (int, optional): number of closest items to select for each item. Defaults to 5.
        name (str, optional): col_name for final list of similar items based on embeddings. Defaults to 'closest_emb'.

    Returns:
        pd.DataFrame: dataframe with: customer_id and name (list of distinct `article_id`)
    """
    transactions.sort_values(by="t_dat", ascending=False, inplace=True)
    transactions = transactions.groupby(["customer_id"]).head(n_last_bought)
    logger.info(f"Selecting latest {n_last_bought} articles for each customer")
    all_items = list(transactions["article_id"].unique())
    logger.info(f"Number of unique articles left: {len(all_items)}")
    closest_dict = _create_embedding_dictionary(all_items, embeddings, k_closest)
    closest_embeddings = transactions[["customer_id", "article_id"]].copy()
    closest_embeddings[name] = closest_embeddings["article_id"].map(closest_dict)
    closest_embeddings = _cleanup_closest_embeddings(closest_embeddings, name)
    return closest_embeddings


# COLLECT ALL
def collect_all_candidates(
    global_articles: pd.DataFrame,
    segment_articles: pd.DataFrame,
    prev_bought_articles: pd.DataFrame,
    prev_bough_prod_name: pd.DataFrame,
    closest_image_embeddings: pd.DataFrame,
    closest_text_embeddings: pd.DataFrame,
) -> pd.DataFrame:
    """Collects candidates from multiple, various methods and joins them together.

    Args:
        global_articles (pd.DataFrame): global articles (most sold globally)
        segment_articles (pd.DataFrame): segment articles (most sold by age segment)
        prev_bought_articles (pd.DataFrame): previously bought articles by each customer
        prev_bough_prod_name (pd.DataFrame):
        previously bought articles from the same `prod_name` for which customer previously bought article from
        closest_image_embeddings (pd.DataFrame):
        most similar items to the ones each customer has bought based on image embeddings
        closest_text_embeddings (pd.DataFrame):
        most similar items to the ones each customer has bought based on text embeddings

    Returns:
        pd.DataFrame: df with: customer_id and column for each candidate generation method with list of `article_id`
    """
    collected_df = (
        global_articles.merge(segment_articles, on="customer_id", how="left")
        .merge(prev_bought_articles, on="customer_id", how="left")
        .merge(prev_bough_prod_name, on="customer_id", how="left")
        .merge(closest_image_embeddings, on="customer_id", how="left")
        .merge(closest_text_embeddings, on="customer_id", how="left")
    )
    return collected_df
