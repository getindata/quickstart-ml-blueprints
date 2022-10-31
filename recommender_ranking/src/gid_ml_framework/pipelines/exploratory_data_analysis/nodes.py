import logging
from datetime import datetime

import mlflow
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas_profiling import ProfileReport

logger = logging.getLogger(__name__)


# auto
def auto_eda(df: pd.DataFrame, name: str) -> None:
    """Automatic exploratory data analysis from pandas_profiling. Saves results
    into mlflow.

    Args:
        df: dataframe on which to run auto EDA
    """
    minimal = True if df.size > 1_000_000 else False
    logger.info(f"Setting {minimal=} for dataframe: {name}")
    profile = ProfileReport(
        df, title=f"Pandas Profiling Report - {name}", minimal=minimal
    )
    profile_html = profile.to_html()
    mlflow.log_text(profile_html, f"eda/auto_eda_{name}.html")


# manual
# visualizations taken from https://www.kaggle.com/code/vanguarde/h-m-eda-first-look
# and https://www.kaggle.com/code/gpreda/h-m-eda-and-prediction
# just as an example
def _garments_grouped_by_index(articles: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(15, 7))
    ax = sns.histplot(
        data=articles,
        y="garment_group_name",
        color="orange",
        hue="index_group_name",
        multiple="stack",
    )
    ax.set_xlabel("count by garment group")
    ax.set_ylabel("garment group")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "eda/manual_garments_per_index.png")
    plt.close()


def _transactions_per_day(transactions: pd.DataFrame) -> None:
    df = (
        transactions.sample(100_000)
        .groupby(["t_dat"])["article_id"]
        .count()
        .reset_index()
    )
    df["t_dat"] = df["t_dat"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    df.columns = ["Date", "Transactions"]
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    plt.plot(df["Date"], df["Transactions"], color="Darkgreen")
    plt.xlabel("Date")
    plt.ylabel("Transactions")
    plt.title("Transactions per day (100k sample)")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "eda/manual_transactions_per_day.png")
    plt.close()


def _price_per_product_groups(
    articles: pd.DataFrame, transactions: pd.DataFrame
) -> None:
    # join
    transactions_articles = transactions[["article_id", "price"]].merge(
        articles[["article_id", "product_group_name", "index_name"]],
        on="article_id",
        how="left",
    )
    # plot
    sns.set_style("darkgrid")
    f, ax = plt.subplots(figsize=(25, 18))
    ax = sns.boxplot(data=transactions_articles, x="price", y="product_group_name")
    ax.set_xlabel("Price outliers", fontsize=22)
    ax.set_ylabel("Index names", fontsize=22)
    ax.xaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_tick_params(labelsize=22)
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "eda/manual_price_outliers_per_prod_group_name.png")
    plt.close()
    # plot 2
    df = (
        transactions_articles[["product_group_name", "price"]]
        .groupby(["product_group_name"])["price"]
        .mean()
        .reset_index()
    )
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.barplot(x=df.price, y=df.product_group_name, color="orange", alpha=0.8)
    ax.set_xlabel("Price by product group")
    ax.set_ylabel("Product group")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "eda/manual_mean_price_by_prod_group_name.png")
    plt.close()
    # plot 3
    df = (
        transactions_articles[["index_name", "price"]]
        .groupby(["index_name"])["price"]
        .mean()
        .reset_index()
    )
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.barplot(x=df.price, y=df.index_name, color="orange", alpha=0.8)
    ax.set_xlabel("Price by index")
    ax.set_ylabel("Index")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "eda/manual_mean_price_by_index_name.png")
    plt.close()


def manual_eda(articles: pd.DataFrame, transactions: pd.DataFrame) -> None:
    _garments_grouped_by_index(articles)
    _price_per_product_groups(articles, transactions)
    _transactions_per_day(transactions)
