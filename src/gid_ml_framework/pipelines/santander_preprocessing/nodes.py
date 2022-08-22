from typing import Iterator, Union, Tuple
from datetime import datetime
import logging
from xmlrpc.client import Boolean

import pandas as pd
import numpy as np
from kedro.extras.datasets.pandas import CSVDataSet
from kedro.io.core import get_filepath_str


pd.options.mode.chained_assignment = None
log = logging.getLogger(__name__)


def _load(self)  -> pd.DataFrame:
    """Modified load function for CSVDataSet since in Kedro version < 0.18
    there is a bug with loading pandas dataset with chunksize parameter for
    bigger datasets.

    Returns:
        pd.DataFrame: loaded data frame
    """
    load_path = get_filepath_str(self._get_load_path(), self._protocol)

    return pd.read_csv(load_path, **self._load_args)


# Overwriting load method because of chunksize bug
CSVDataSet._load = _load


def sample_santander(santander: Iterator[pd.DataFrame],
                     sample_user_frac: float=0.1,
                     cutoff_date: Union[str, datetime]='2016-05-28') \
                     -> pd.DataFrame:
    """Sample Santader data based on user sample size and cutoff date.

    Args:
        santander (Iterator[pd.DataFrame]): raw data chunks
        sample_user (float): fraction of users
        cutoff_date (Union[str, datetime]): filtering date point

    Returns:
        pd.DataFrame: data sample
    """
    santander_df = pd.DataFrame()
    for chunk in santander:
        santander_df = pd.concat([santander_df, chunk], ignore_index=True)
    log.info(f"Santander df shape before sampling: {santander_df.shape}")
    if np.isclose(sample_user_frac, 1.0):
        santander_sample = santander_df
    else:
        unique_ids   = pd.Series(santander_df["ncodpers"].unique())
        users_limit = int(len(unique_ids)*sample_user_frac)
        unique_ids = unique_ids.sample(n=users_limit)
        santander_sample = (santander_df[santander_df['ncodpers']
                            .isin(unique_ids)])
    santander_sample.loc[:, 'fecha_dato'] = (pd.to_datetime(santander_sample
                                             .loc[:, 'fecha_dato']))
    santander_sample = (santander_sample[santander_sample['fecha_dato']
                        <= cutoff_date])
    log.info(f"Santander df shape after sampling: {santander_sample.shape}")
    return santander_sample


def filter_santander(santander_df: Iterator[pd.DataFrame]) -> pd.DataFrame:
    """Filter unused data and columns

    Args:
        santander_df (Iterator[pd.DataFrame]): santander input dataframe

    Returns:
        pd.DataFrame: filtered dataframe
    """
    df = pd.DataFrame()
    for chunk in santander_df:
        df = pd.concat([df, chunk], ignore_index=True)
    log.info(f"Santander df shape before filtering: {df.shape}")
    # Information already present in other columns. Name of the province exists
    # in nomprov.
    df.drop(["tipodom", "cod_prov"], axis=1,inplace=True)
    log.info(f"Santander df shape after filtering: {df.shape}")
    return df


def clean_santander(santander_df: Iterator[pd.DataFrame]) -> pd.DataFrame:
    """Basic preprocessing steps for input dataframe

    Args:
        santander_df (Iterator[pd.DataFrame]): santander input dataframe

    Returns:
        pd.DataFrame: preprocessed dataframe
    """
    df = pd.DataFrame()
    for chunk in santander_df:
        df = pd.concat([df, chunk], ignore_index=True)
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
    df = pd.DataFrame()
    for chunk in santander_sample:
        df = pd.concat([df, chunk], ignore_index=True)
    log.info(f'Dataframe size before splitting: {df.shape}')
    last_month = df['fecha_dato'].max()
    log.info(f'Dataframe last month: {last_month}')
    train_df = df[df['fecha_dato'] < last_month]
    val_df = df[df['fecha_dato'] >= last_month]
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
    median = df['renta'].median(skipna=True)
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
    df = pd.DataFrame()
    for chunk in santander_df:
        df = pd.concat([df, chunk], ignore_index=True)
    log.info(f'Number of columns with missing values before imputing: \
    {df.isnull().any().sum()}')

    df.loc[:, 'month'] = pd.DatetimeIndex(df['fecha_dato']).month
    df.loc[:, 'age'] = pd.to_numeric(df['age'], errors='coerce')
    # Age imputing
    df.loc[df['age'] < 18, "age"]  = (df.loc[(df['age'] >= 18)
                                      & (df['age'] <= 30), "age"]
                                      .mean(skipna=True))
    df.loc[df['age'] > 100, "age"] = (df.loc[(df['age'] >= 30) 
                                      & (df['age'] <= 100), "age"]
                                      .mean(skipna=True))
    df["age"].fillna(df["age"].mean(), inplace=True)
    df.loc[:, 'age'] = df["age"].astype(int)
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
    df.loc[df['nomprov'].isnull(),"nomprov"] = "UNKNOWN"
    # Transforming string to NA in income column
    mask = df['renta'].map(lambda x: isinstance(x, (int, float)))
    df.loc[:, 'renta'] = df.loc[:, 'renta'].where(mask)
    # Imputing missing gross income values with median of province
    df = df.groupby('nomprov').apply(_median_gross)  
    # If any rows still null (province has all null) replace by overall median
    df.loc[df['renta'].isnull(), "renta"] = df.renta.median
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

    log.info(f'Number of columns with missing values after imputing: \
    {df.isnull().any().sum()}')
    return df


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


def target_processing_santander(input_train_df: Iterator[pd.DataFrame],
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
    train_df = pd.DataFrame()
    for chunk in input_train_df:
        df = pd.concat([df, chunk], ignore_index=True)
    val_df = pd.DataFrame()
    for chunk in input_val_df:
        df = pd.concat([df, chunk], ignore_index=True)
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