import logging
import re
from datetime import datetime
from typing import Iterator, List, Tuple, Union
from xmlrpc.client import Boolean

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from recommender_gnn.extras.datasets.chunks_dataset import _concat_chunks

pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)


def _stratify(df: pd.DataFrame, customers_limit: float) -> List:
    """Stratify customers based on age (age bins),
    tiprel_1mes (Customer relation type indicating if he is active), renta
    (customer income bins) columns values for customers from last month as
    they are also present in test subset

    Args:
        df (pd.DataFrame): raw Santader dataframe
        sample_customer_frac (float): number of customers to sample

    Returns:
        List: stratified sample of customers
    """
    customer_col = "ncodpers"
    age_col = "age"
    relation_col = "tiprel_1mes"
    income_col = "renta"
    # Stratification based on last possible date column values
    df = (
        df.sort_values(by="fecha_dato")
        .reset_index(drop=True)
        .drop_duplicates(customer_col, keep="last")
    )
    strat_cols = [age_col, relation_col, income_col]
    df = df.loc[:, strat_cols + [customer_col]]
    # Age to bins
    df.loc[df[age_col] == " NA", age_col] = None
    df[age_col].fillna(df[age_col].median(), inplace=True)
    df.loc[:, age_col] = df.loc[:, age_col].astype(int)
    df.loc[:, age_col] = pd.qcut(df[age_col], q=3, duplicates="drop").astype("category")
    # Renta to bins
    df.loc[:, income_col] = pd.to_numeric(df.loc[:, income_col], errors="coerce")
    df.loc[df[income_col].isnull(), income_col] = df.renta.median()
    df.loc[:, income_col] = pd.qcut(df[income_col], q=3, duplicates="drop").astype(
        "category"
    )
    # Tiprel_1mes imputing
    df.loc[df[relation_col].isnull(), relation_col] = "A"
    df.loc[df[relation_col].isin(["P", "R"]), relation_col] = "I"
    df.loc[:, relation_col] = df[relation_col].astype("category")

    sample_frac = float(customers_limit) / float(df.shape[0])
    _, customers_sample = train_test_split(
        df, test_size=sample_frac, stratify=df.loc[:, ["age"]]
    )
    customers_sample.drop(strat_cols, axis=1, inplace=True)
    customers_list = np.unique(customers_sample.values.tolist())
    return customers_list


def sample_santander(
    santander: Iterator[pd.DataFrame],
    sample_customer_frac: float = 0.1,
    cutoff_date: Union[str, datetime] = "2016-05-28",
    stratify: Boolean = False,
) -> pd.DataFrame:
    """Sample Santader data based on customer sample size and cutoff date.

    Args:
        santander (Iterator[pd.DataFrame]): raw data chunks
        sample_customer_frac (float): fraction of customers
        cutoff_date (Union[str, datetime]): filtering date point
        stratify (Boolean): should sample be stratified based on age, income
            and customer relation type

    Returns:
        pd.DataFrame: data sample
    """
    santander_df = _concat_chunks(santander)
    logger.info(f"Santander df shape before sampling: {santander_df.shape}")
    last_possible_date = pd.to_datetime("2016-05-28")
    santander_df.loc[:, "fecha_dato"] = pd.to_datetime(
        santander_df.loc[:, "fecha_dato"]
    )
    cutoff_date = pd.to_datetime(cutoff_date)
    # Cut off data based on given date
    if cutoff_date < last_possible_date:
        santander_df = santander_df[santander_df["fecha_dato"] <= cutoff_date]
    if np.isclose(sample_customer_frac, 1.0):
        santander_sample = santander_df
    else:
        unique_ids = pd.Series(santander_df["ncodpers"].unique())
        customers_limit = int(len(unique_ids) * sample_customer_frac)
        if not customers_limit:
            return pd.DataFrame({})
        if stratify:
            unique_ids = _stratify(santander_df, customers_limit)
        else:
            unique_ids = pd.Series(santander_df["ncodpers"].unique())
            customers_limit = int(len(unique_ids) * sample_customer_frac)
            unique_ids = unique_ids.sample(n=customers_limit)
        santander_sample = santander_df[santander_df["ncodpers"].isin(unique_ids)]
    logger.info(f"Santander df shape after sampling: {santander_sample.shape}")
    return santander_sample


def filter_santander(santander_df: Iterator[pd.DataFrame]) -> pd.DataFrame:
    """Filter unused data and columns

    Args:
        santander_df (Iterator[pd.DataFrame]): santander input dataframe

    Returns:
        pd.DataFrame: filtered dataframe
    """
    df = _concat_chunks(santander_df)
    logger.info(f"Santander df shape before filtering: {df.shape}")
    # Information already present in other columns. Name of the province exists
    # in nomprov.
    df.drop(["tipodom", "cod_prov"], axis=1, inplace=True)
    logger.info(f"Santander df shape after filtering: {df.shape}")
    return df


def clean_santander(santander_df: Iterator[pd.DataFrame]) -> pd.DataFrame:
    """Basic preprocessing steps for input dataframe

    Args:
        santander_df (Iterator[pd.DataFrame]): santander input dataframe

    Returns:
        pd.DataFrame: preprocessed dataframe
    """
    df = _concat_chunks(santander_df)
    # Bad encoding of spain letters
    df.loc[df["nomprov"] == "CORU\xc3\x91A, A", "nomprov"] = "CORUNA, A"
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
    logger.info(f"Dataframe size before splitting: {df.shape}")
    last_month = df["fecha_dato"].max()
    logger.info(f"Dataframe last month: {last_month}")
    train_df = df.loc[df["fecha_dato"] < last_month, :]
    val_df = df.loc[df["fecha_dato"] >= last_month, :]
    logger.info(
        f"Training dataframe size: {train_df.shape}, \
             Validation dataframe size: {val_df.shape}"
    )
    logger.info(
        f"Training dataframe min date: {train_df['fecha_dato'].min()}, \
             max date: {train_df['fecha_dato'].max()}"
    )
    logger.info(
        f"Validation dataframe min date: {val_df['fecha_dato'].min()}, \
             max date: {val_df['fecha_dato'].max()}"
    )
    return (train_df, val_df)


def _impute_income_median(df: pd.DataFrame, income_column: str) -> pd.DataFrame:
    """Auxiliary function which calculates median gross income for data
    imputing

    Args:
        df (pd.DataFrame): dataframe with "renta" columns

    Returns:
        pd.DataFrame: dataframe with calculated median of gross income
    """
    median = df.loc[:, income_column].median(skipna=True)
    df.loc[df[income_column].isnull(), income_column] = median
    return df


def _impute_age(df: pd.DataFrame, age_column: str) -> pd.DataFrame:
    """Imputes column with age of customers"""
    df.loc[:, age_column] = pd.to_numeric(df[age_column], errors="coerce")
    # Age imputing for outliers which are probably errors
    df.loc[df[age_column] < 18, age_column] = df.loc[
        (df[age_column] >= 18) & (df[age_column] <= 30), age_column
    ].mean(skipna=True)
    df.loc[df[age_column] > 100, age_column] = df.loc[
        (df[age_column] >= 30) & (df[age_column] <= 100), age_column
    ].mean(skipna=True)
    # Mean imputing for the rest
    df[age_column].fillna(df[age_column].mean(), inplace=True)
    df.loc[:, age_column] = df.loc[:, age_column].astype(int)
    return df


def _impute_seniority(df: pd.DataFrame, seniority_column: str) -> pd.DataFrame:
    """Imputes column which indicates customer seniority (in months)"""
    df.loc[:, seniority_column] = pd.to_numeric(df[seniority_column], errors="coerce")
    # Imputing seniority level for probably new customers
    df.loc[df[seniority_column].isnull(), seniority_column] = df[seniority_column].min()
    df.loc[df[seniority_column] < 0, seniority_column] = 0
    return df


def _impute_joining_date(df: pd.DataFrame, joining_date_column: str) -> pd.DataFrame:
    """Imputes column with the date in which the customer became as the first holder of a contract in the bank"""
    # Imputing date of joining the bank with median date
    dates = df.loc[:, joining_date_column].sort_values().reset_index()
    median_date = int(np.median(dates.index.values))
    df.loc[df[joining_date_column].isnull(), joining_date_column] = dates.loc[
        median_date, joining_date_column
    ]
    return df


def _impute_income(df: pd.DataFrame, income_column: str) -> pd.DataFrame:
    """ "Imputes values in column with gross income of the household"""
    # Transforming string to NA in income column
    df.loc[:, income_column] = pd.to_numeric(df.loc[:, income_column], errors="coerce")
    # Imputing missing gross income values with median of province
    df = df.groupby("nomprov").apply(lambda x: _impute_income_median(x, income_column))
    # If any rows still null (province has all null) replace by overall median
    df.loc[df[income_column].isnull(), income_column] = df.loc[
        :, income_column
    ].median()
    df = df.sort_values(by="fecha_dato").reset_index(drop=True)
    return df


def _impute_cutomer_type(df: pd.DataFrame, customer_type_column: str) -> pd.DataFrame:
    """Imputes column with values for customer type at the beginning of the month;
    1 - (First/Primary customer), 2 - (co-owner ), P - (Potential), 3 - (former primary), 4 - (former co-owner)
    """

    map_dict = {
        1.0: "1",
        "1.0": "1",
        "1": "1",
        "3.0": "3",
        "P": "P",
        3.0: "3",
        2.0: "2",
        "3": "3",
        "2.0": "2",
        "4.0": "4",
        "4": "4",
        "2": "2",
    }

    # Imputing with dataset mode and applying custom mapping
    df[customer_type_column].fillna("P", inplace=True)
    df.loc[:, customer_type_column] = df[customer_type_column].apply(
        lambda x: map_dict.get(x, x)
    )
    df.loc[:, customer_type_column] = df[customer_type_column].astype("category")
    return df


def _impute_new_category(df: pd.DataFrame) -> pd.DataFrame:
    """Impute rest of missing values in categorical data with new category"""
    string_data = df.select_dtypes(include=["object"])
    missing_columns = [col for col in string_data if string_data[col].isnull().any()]
    if "conyuemp" not in missing_columns:
        missing_columns += ["conyuemp"]
    unknown_cols = [
        col
        for col in missing_columns
        if col not in ["indfall", "tiprel_1mes", "indrel_1mes"]
    ]
    for col in unknown_cols:
        df.loc[df[col].isnull(), col] = "UNKNOWN"
    del string_data
    return df


def _impute_products(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes bank products columns"""
    # Imputing product feature values with median - "not owned" flag, because of small number of missing values in total
    # and that most of missing values are for new customers
    df.loc[df["ind_nomina_ult1"].isnull(), "ind_nomina_ult1"] = 0
    df.loc[df["ind_nom_pens_ult1"].isnull(), "ind_nom_pens_ult1"] = 0
    r = re.compile("ind_+.*ult.*")
    feature_cols = list(filter(r.match, df.columns))
    for col in feature_cols:
        df.loc[:, col] = df[col].astype(int)
    return df


def _impute_customer_relation(
    df: pd.DataFrame, customer_relation_column: str
) -> pd.DataFrame:
    """Imputes columns with customer relation type at the beginning of the month, A - (active), I -(inactive),
    P - (former customer), R - (Potential)
    """
    # Imputing customer relation with overall mode - active client
    df.loc[df[customer_relation_column].isnull(), customer_relation_column] = "A"
    df.loc[:, customer_relation_column] = df[customer_relation_column].astype(
        "category"
    )
    return df


def _convert_int_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Change type of columns that should be int"""
    int_cols = ["ind_nuevo", "antiguedad", "indrel", "ind_actividad_cliente"]
    for col in int_cols:
        df.loc[:, col] = df[col].astype(int)
    return df


def impute_santander(
    santander_df: Iterator[pd.DataFrame], test: Boolean = False
) -> pd.DataFrame:
    """Impute missing values in splitted Santander dataframe based on
    information from tested Kaggle submissions.

    Args:
        santander_df (Iterator[pd.DataFrame]): train/val santander input data

    Returns:
        pd.DataFrame: imputed dataframe
    """
    df = _concat_chunks(santander_df)
    logger.info(
        f"Number of columns with missing values before imputing: {df.isnull().any().sum()}"
    )

    df = _impute_age(df, age_column="age")
    # Imputing new customer flag, as missing values are present for new customers
    df.loc[df["ind_nuevo"].isnull(), "ind_nuevo"] = 1
    df = _impute_seniority(df, seniority_column="antiguedad")
    df = _impute_joining_date(df, joining_date_column="fecha_alta")
    # Imputing "primary" customer level flag with mode
    df.loc[df["indrel"].isnull(), "indrel"] = 1
    # Imputing active client flag with median
    df.loc[df["ind_actividad_cliente"].isnull(), "ind_actividad_cliente"] = df.loc[
        :, "ind_actividad_cliente"
    ].median()
    # Imputing missing province name with new category
    df.loc[df["nomprov"].isnull(), "nomprov"] = "UNKNOWN"
    df = _impute_income(df, income_column="renta")
    if not test:
        df = _impute_products(df)
    # Missing values for dead customers imputed as Negative - (mode)
    df.loc[df["indfall"].isnull(), "indfall"] = "N"
    df = _impute_customer_relation(df, customer_relation_column="tiprel_1mes")
    df = _impute_cutomer_type(df, customer_type_column="indrel_1mes")
    df = _impute_new_category(df)
    df = _convert_int_columns(df)

    logger.info(
        f"Number of columns with missing values after imputing: {df.isnull().any().sum()}"
    )
    logger.info(
        f"Columns with missing values: {df.columns[df.isnull().any()].tolist()}"
    )
    return df
