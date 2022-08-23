from typing import Iterator, Tuple
import logging
import re

import pandas as pd
import numpy as np
from kedro.extras.datasets.pandas import CSVDataSet

from gid_ml_framework.extras.datasets.chunks_dataset import (
 _load,
 _concat_chunks,
)


pd.options.mode.chained_assignment = None
log = logging.getLogger(__name__)
# Overwriting load method because of chunksize bug in Kedro < 0.18
CSVDataSet._load = _load


def santander_to_articles(santander: Iterator[pd.DataFrame]) \
    -> Tuple[pd.DataFrame, pd.DataFrame]:
    """From Santander train/val dataset extract articles data

    Args:
        santander (pd.DataFrame): preprocessed train/val santander dataset

    Returns:
        pd.Series: list of articles ids in Santander dataset
        pd.DataFrame: dataframe with features of each article
    """
    r = re.compile("ind_+.*ult.*")
    articles = pd.DataFrame({'articles': list(filter(r.match,
                                                     santander.columns))})
    # There is no article features in Santander dataset                                               
    articles_features = pd.DataFrame()
    return (articles, articles_features)



def santander_to_customers(santander: Iterator[pd.DataFrame],
                           merge_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """From Santander train/val/test dataset extract customers data

    Args:
        santander (pd.DataFrame): preprocessed train/val santander dataset
        merge_type (str): function which will be used to merge customer
            features from all timestamps into one representation

    Returns:
        pd.Series: list of customers ids in Santander dataset
        pd.DataFrame: dataframe with features of each customer
    """
    

def _status_change(x: pd.Series) -> str:
    """Based on difference of the following rows create label which indicates
    if given product was added, dropped or maintained in comparison with last 
    month for given customer.

    Args:
        x (pd.DataFrame): imputed santander train dataframe

    Returns:
        str: target label - added/dropped/maintained
    """
    diffs = x.diff().fillna(0)
    # First occurrence is considered as "Maintained"
    label = ["Added" if i == 1 \
         else "Dropped" if i == -1 \
         else "Maintained" for i in diffs]
    return label


def _target_processing_santander(input_train_df: Iterator[pd.DataFrame],
                                input_val_df: Iterator[pd.DataFrame]) -> Tuple:
    """Preprocess target columns to focus on products that will be bought
    in the next month

    Args:
        input_train_df (Iterator[pd.DataFrame]): imputed santander train
        dataframe
        input_val_df (Iterator[pd.DataFrame]): imputed santander validation 
        dataframe

    Returns:
        Tuple: processed train and validation dataframes
    """
    train_df = _concat_chunks(input_train_df)
    val_df = _concat_chunks(input_val_df)
    train_len = len(train_df)
    df = pd.concat([train_df, val_df])
    feature_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
    # Create auxiliary column
    unique_months = (pd.DataFrame(pd.Series(df.fecha_dato.unique())
                     .sort_values()).reset_index(drop=True))
    # Start with month 1, not 0 to match what we already have
    unique_months["month_id"] = pd.Series(range(1, 1+unique_months.size))
    unique_months["month_next_id"] = 1 + unique_months["month_id"]
    unique_months.rename(columns={0:"fecha_dato"}, inplace=True)
    df = pd.merge(df,unique_months, on="fecha_dato")
    # Apply status change labeling
    df.loc[:, feature_cols] = (df.loc[:, [i for i in feature_cols]
                               + ["ncodpers"]].groupby("ncodpers")
                               .transform(_status_change))
    # Can be done faster but some tweaks needed 
    # df = df.sort_values(['ncodpers', 'fecha_dato']).reset_index(drop=True)
    # s = df.loc[:, [i for i in feature_cols] + ["ncodpers"]]
    # df.loc[:, feature_cols] = (s.loc[:, [i for i in feature_cols]].diff()
    #                            .where(s.duplicated(["ncodpers"],
    #                            keep='first'))
    #                            ).fillna(0).transform(_status_change2)
    df = df.sort_values(['fecha_dato']).reset_index(drop=True)

    log.info(f'Sum of number of newly added products: \
             {df.eq("Added").sum().sum()}')

    train_df = df.iloc[:train_len, :]
    val_df = df.iloc[train_len:, :]
    return (train_df, val_df)


def santander_to_transactions(santander: Iterator[pd.DataFrame]) \
    -> Tuple[pd.DataFrame, pd.DataFrame]:
    """From Santander train/val datasets extract transactions data

    Args:
        santander (pd.DataFrame): preprocessed train/val santander dataset

    Returns:
        pd.Series: list of transactions ids in Santander dataset
        pd.DataFrame: dataframe with data about each transaction
    """