from typing import Iterator, Union, Tuple, List
from datetime import datetime
import logging
from xmlrpc.client import Boolean

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from kedro.extras.datasets.pandas import CSVDataSet

from gid_ml_framework.extras.datasets.chunks_dataset import (
 _load,
 _concat_chunks,
)


pd.options.mode.chained_assignment = None
log = logging.getLogger(__name__)
# Overwriting load method because of chunksize bug in Kedro < 0.18
CSVDataSet._load = _load


def _stratify(df: pd.DataFrame, sample_customer_frac: float,
              cutoff_date: Union[str, datetime]) -> List:
    """Stratify customers based on age (age bins), 
    tiprel_1mes (Customer relation type indicating if he is active), renta 
    (customer income bins) columns values from last month

    Args:
        df (Iterator[pd.DataFrame]): raw Santader dataframe
        sample_customer_frac (float): fraction of customers
        cutoff_date (Union[str, datetime]): filtering date point

    Returns:
        List: stratified sample of customers
    """
    # Stratification based on last month column values
    df = df.loc[df['fecha_dato'] == cutoff_date, :]
    strat_cols = ['age', 'tiprel_1mes', 'renta']
    df = df.loc[:, strat_cols + ['ncodpers']]
    # Age to bins
    df["age"].fillna(df["age"].median(), inplace=True)
    df.loc[:, 'age'] = df.loc[:, "age"].astype(int)
    df.loc[:, 'age'] = pd.qcut(df['age'], q=3,
                               labels=['1', '2', '3']).astype("category")
    # Renta to bins
    df.loc[:, 'renta'] = pd.to_numeric(df.loc[:, 'renta'],
                                       errors='coerce')
    df.loc[df['renta'].isnull(), "renta"] = df.renta.median()
    df.loc[:, 'renta'] = pd.qcut(df['renta'], q=3,
                               labels=['1', '2', '3']).astype("category")
    # Tiprel_1mes imputing
    df.loc[df['tiprel_1mes'].isnull(), "tiprel_1mes"] = "A"
    df.loc[df['tiprel_1mes'].isin(['P', 'R']), "tiprel_1mes"] = "I"
    df.loc[:, 'tiprel_1mes'] = df['tiprel_1mes'].astype("category")

    _, customers_sample = train_test_split(
        df,
        test_size=sample_customer_frac, 
        stratify=df.loc[:, strat_cols]
    )
    customers_sample.drop(strat_cols, axis=1, inplace=True)
    customers_list = np.unique(customers_sample.values.tolist())
    return customers_list


def sample_santander(santander: Iterator[pd.DataFrame],
                     sample_customer_frac: float=0.1,
                     cutoff_date: Union[str, datetime]='2016-05-28',
                     stratify: Boolean=False) \
                     -> pd.DataFrame:
    """Sample Santader data based on customer sample size and cutoff date.

    Args:
        santander (Iterator[pd.DataFrame]): raw data chunks
        sample_customer_frac (float): fraction of customers
        cutoff_date (Union[str, datetime]): filtering date point

    Returns:
        pd.DataFrame: data sample
    """
    santander_df = _concat_chunks(santander)
    log.info(f"Santander df shape before sampling: {santander_df.shape}")
    santander_df.loc[:, 'fecha_dato'] = (pd.to_datetime(santander_df
                                             .loc[:, 'fecha_dato']))
    if cutoff_date != '2016-05-28':
        santander_df = (santander_df[santander_df['fecha_dato']
                            <= cutoff_date])
    if np.isclose(sample_customer_frac, 1.0):
        santander_sample = santander_df
    else:
        if stratify:
            unique_ids = _stratify(santander_df, sample_customer_frac,
                                   cutoff_date)
        else:
            unique_ids = pd.Series(santander_df["ncodpers"].unique())
            customers_limit = int(len(unique_ids) * sample_customer_frac)
            unique_ids = unique_ids.sample(n=customers_limit)
        santander_sample = (santander_df[santander_df['ncodpers']
                            .isin(unique_ids)])
    log.info(f"Santander df shape after sampling: {santander_sample.shape}")
    return santander_sample


def filter_santander(santander_df: Iterator[pd.DataFrame]) -> pd.DataFrame:
    """Filter unused data and columns

    Args:
        santander_df (Iterator[pd.DataFrame]): santander input dataframe

    Returns:
        pd.DataFrame: filtered dataframe
    """
    df = _concat_chunks(santander_df)
    log.info(f"Santander df shape before filtering: {df.shape}")
    # Information already present in other columns. Name of the province exists
    # in nomprov.
    df.drop(["tipodom", "cod_prov"], axis=1, inplace=True)
    log.info(f"Santander df shape after filtering: {df.shape}")
    return df


def clean_santander(santander_df: Iterator[pd.DataFrame]) -> pd.DataFrame:
    """Basic preprocessing steps for input dataframe

    Args:
        santander_df (Iterator[pd.DataFrame]): santander input dataframe

    Returns:
        pd.DataFrame: preprocessed dataframe
    """
    df = _concat_chunks(santander_df)
    # Bad encoding of spain letter
    df.loc[df['nomprov'] == "CORU\xc3\x91A, A", "nomprov"] = "CORUNA, A"
    return df


def split_santander(santander_sample: Iterator[pd.DataFrame]) -> Tuple:
    """Split input dataframe into train and val splits. Validation part
    consists of last month and train part consists of remaining months.

    Args:
        santander_sample (Iterator[pd.DataFrame]): santander input dataframe
        date_column (str): date column based on which split will be performed

    Returns:
        Tuple: val and train dataframes 
    """
    df = _concat_chunks(santander_sample)
    log.info(f'Dataframe size before splitting: {df.shape}')
    last_month = df['fecha_dato'].max()
    log.info(f'Dataframe last month: {last_month}')
    train_df = df.loc[df['fecha_dato'] < last_month, :]
    val_df = df.loc[df['fecha_dato'] >= last_month, :]
    log.info(f'Training dataframe size: {train_df.shape}, \
             Validation dataframe size: {val_df.shape}')
    log.info(f'Training dataframe min date: {train_df["fecha_dato"].min()}, \
             max date: {train_df["fecha_dato"].max()}')
    log.info(f'Validation dataframe min date: {val_df["fecha_dato"].min()}, \
             max date: {val_df["fecha_dato"].max()}')
    return (train_df, val_df)


def _median_gross(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates median gross income for each province for imputing data

    Args:
        df (pd.DataFrame): dataframe with 'renta' columns

    Returns:
        pd.DataFrame: dataframe with calculated median of gross income
    """
    median = df.loc[:, 'renta'].median(skipna=True)
    df.loc[df['renta'].isnull(), 'renta'] = median
    return df


def impute_santander(santander_df: Iterator[pd.DataFrame],
                     test: Boolean=False) -> pd.DataFrame:
    """Impute missing values in splitted Santander dataframe.

    Args:
        santander_df (Iterator[pd.DataFrame]): train/val santander input data

    Returns:
        pd.DataFrame: imputed dataframe
    """
    df = _concat_chunks(santander_df)
    log.info(f'Number of columns with missing values before imputing: \
    {df.isnull().any().sum()}')

    df.loc[:, 'age'] = pd.to_numeric(df['age'], errors='coerce')
    # Age imputing
    df.loc[df['age'] < 18, "age"]  = (df.loc[(df['age'] >= 18)
                                      & (df['age'] <= 30), "age"]
                                      .mean(skipna=True))
    df.loc[df['age'] > 100, "age"] = (df.loc[(df['age'] >= 30) 
                                      & (df['age'] <= 100), "age"]
                                      .mean(skipna=True))
    df["age"].fillna(df["age"].mean(), inplace=True)
    df.loc[:, 'age'] = df.loc[:, "age"].astype(int)
    # Imputing new customer flag, as missing values are present for new customers
    df.loc[df["ind_nuevo"].isnull(),"ind_nuevo"] = 1
    # Imputing seniority level for these new customers
    df.loc[:, 'antiguedad'] = pd.to_numeric(df['antiguedad'], errors="coerce")
    df.loc[df['antiguedad'].isnull(), "antiguedad"] = df['antiguedad'].min()
    df.loc[df['antiguedad'] < 0, "antiguedad"] = 0
    # Imputing date of joining the bank with median
    dates=df.loc[:, "fecha_alta"].sort_values().reset_index()
    median_date = int(np.median(dates.index.values))
    df.loc[df["fecha_alta"].isnull(), "fecha_alta"] = dates.loc[median_date,
                                                                "fecha_alta"]
    # Imputing "primary" customer level flag with mode
    df.loc[df['indrel'].isnull(), "indrel"] = 1
    # Imputing active client flag
    df.loc[df['ind_actividad_cliente'].isnull(),"ind_actividad_cliente"] = \
    df.loc[:, "ind_actividad_cliente"].median()
    # Imputing missing province name
    df.loc[df['nomprov'].isnull(), "nomprov"] = "UNKNOWN"
    # Transforming string to NA in income column
    df.loc[:, 'renta'] = pd.to_numeric(df.loc[:, 'renta'],
                                       errors='coerce')
    # Imputing missing gross income values with median of province
    df = df.groupby('nomprov').apply(_median_gross) 
    # If any rows still null (province has all null) replace by overall median
    df.loc[df['renta'].isnull(), "renta"] = df.renta.median() 
    df = df.sort_values(by="fecha_dato").reset_index(drop=True)
    # Imputing product values with median, because of small number
    if not test:
        df.loc[df['ind_nomina_ult1'].isnull(), "ind_nomina_ult1"] = 0
        df.loc[df['ind_nom_pens_ult1'].isnull(), "ind_nom_pens_ult1"] = 0
    # Imputing string columns with empty characters with median or new category
    df.loc[df['indfall'].isnull(), "indfall"] = "N"
    df.loc[df['tiprel_1mes'].isnull(), "tiprel_1mes"] = "A"
    df.loc[:, 'tiprel_1mes'] = df['tiprel_1mes'].astype("category")

    map_dict = { 1.0  : "1",
            "1.0" : "1",
            "1"   : "1",
            "3.0" : "3",
            "P"   : "P",
            3.0   : "3",
            2.0   : "2",
            "3"   : "3",
            "2.0" : "2",
            "4.0" : "4",
            "4"   : "4",
            "2"   : "2"}

    df['indrel_1mes'].fillna("P", inplace=True)
    df.loc[:, 'indrel_1mes'] = df['indrel_1mes'].apply(lambda x:
                                                       map_dict.get(x, x))
    df.loc[:, 'indrel_1mes'] = df['indrel_1mes'].astype("category")

    string_data = df.select_dtypes(include=["object"])
    missing_columns = ([col for col in string_data if string_data[col]
                        .isnull().any()])
    if 'conyuemp' not in missing_columns:
        missing_columns += ['conyuemp']
    unknown_cols = [col for col in missing_columns if col not in
                    ["indfall", "tiprel_1mes", "indrel_1mes"]]
    for col in unknown_cols:
        df.loc[df[col].isnull(), col] = "UNKNOWN"
    del string_data
    # Convert product features to integer values
    if not test:
        feature_cols = (df.iloc[:1, ].filter(regex="ind_+.*ult.*")
                        .columns.values)
        for col in feature_cols:
            df.loc[:, col] = df[col].astype(int)
    int_cols = ['ind_nuevo', 'antiguedad', 'indrel', 'ind_actividad_cliente']
    for col in int_cols:
            df.loc[:, col] = df[col].astype(int)
    log.info(f'Number of columns with missing values after imputing: \
    {df.isnull().any().sum()}')
    log.info(f'Columns with missing values: \
            {df.columns[df.isnull().any()].tolist()}')
    return df