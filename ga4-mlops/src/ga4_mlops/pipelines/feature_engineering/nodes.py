"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.4
"""
import logging
import re

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    logger.info("Applying manual feature engineering transformations...")

    df["c_weekday"] = pd.to_datetime(df["i_visit_start_time"], unit="us").dt.weekday
    df["c_visit_start_hour"] = pd.to_datetime(
        df["i_visit_start_time"], unit="us"
    ).dt.hour

    return df


def fit_imputers(df: pd.DataFrame, imputation_strategies: dict) -> dict:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        imputation_strategies (dict): _description_

    Returns:
        pd.DataFrame: _description_
    """
    logger.info("Fitting missing values imputers...")
    columns_to_impute = [
        item for sublist in list(imputation_strategies.values()) for item in sublist
    ]
    assert all(
        item in df.columns for item in columns_to_impute
    ), "Some of columns to encode are not in df"

    num_cols = [item for item in df.columns if re.compile("^n_").match(item)]
    cat_cols = [item for item in df.columns if re.compile("^c_").match(item)]

    df[num_cols] = df[num_cols].astype(float)
    df[cat_cols] = np.where(
        pd.isnull(df[cat_cols]), df[cat_cols], df[cat_cols].astype(str)
    )

    if len(imputation_strategies["mean"]) > 0:
        num_mean_imputer = SimpleImputer(strategy="mean")
        num_mean_imputer.fit(df[imputation_strategies["mean"]])
    else:
        num_mean_imputer = None

    if len(imputation_strategies["zero"]) > 0:
        num_zero_imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        num_zero_imputer.fit(df[imputation_strategies["zero"]])
    else:
        num_zero_imputer = None

    if len(imputation_strategies["mostfreq"]) > 0:
        cat_mostfreq_imputer = SimpleImputer(strategy="most_frequent")
        cat_mostfreq_imputer.fit(df[imputation_strategies["mostfreq"]])
    else:
        cat_mostfreq_imputer = None

    if len(imputation_strategies["unknown"]) > 0:
        cat_unknown_imputer = SimpleImputer(strategy="constant", fill_value="UNKNOWN")
        cat_unknown_imputer.fit(df[imputation_strategies["unknown"]])
    else:
        cat_unknown_imputer = None

    imputers = {
        "mean": num_mean_imputer,
        "zero": num_zero_imputer,
        "mostfreq": cat_mostfreq_imputer,
        "unknown": cat_unknown_imputer,
    }

    return imputers


def apply_imputers(df: pd.DataFrame, imputers: dict) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        encoders (dict): _description_

    Returns:
        pd.DataFrame: _description_
    """
    logger.info("Applying missing values imputers...")

    num_cols = [item for item in df.columns if re.compile("^n_").match(item)]
    cat_cols = [item for item in df.columns if re.compile("^c_").match(item)]

    df[num_cols] = df[num_cols].astype(float)
    df[cat_cols] = np.where(
        pd.isnull(df[cat_cols]), df[cat_cols], df[cat_cols].astype(str)
    )

    if imputers["mean"] is not None:
        df[imputers["mean"].feature_names_in_] = imputers["mean"].transform(
            df[imputers["mean"].feature_names_in_]
        )

    if imputers["zero"] is not None:
        df[imputers["zero"].feature_names_in_] = imputers["zero"].transform(
            df[imputers["zero"].feature_names_in_]
        )

    if imputers["mostfreq"] is not None:
        df[imputers["mostfreq"].feature_names_in_] = imputers["mostfreq"].transform(
            df[imputers["mostfreq"].feature_names_in_]
        )

    if imputers["unknown"] is not None:
        df[imputers["unknown"].feature_names_in_] = imputers["unknown"].transform(
            df[imputers["unknown"].feature_names_in_]
        )

    return df


def fit_encoders(df: pd.DataFrame, encoder_types: dict) -> dict:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    logger.info("Fitting categorical encoders...")
    columns_to_encode = [
        item for sublist in list(encoder_types.values()) for item in sublist
    ]
    assert all(
        item in df.columns for item in columns_to_encode
    ), "Some of columns to encode are not in df"

    binary_encoder = ce.BinaryEncoder(cols=encoder_types["binary"])
    onehot_encoder = ce.OneHotEncoder(cols=encoder_types["onehot"], use_cat_names=True)
    ordinal_encoder = ce.OrdinalEncoder(cols=encoder_types["ordinal"])

    binary_encoder.fit(df)
    df = binary_encoder.transform(df)
    onehot_encoder.fit(df)
    df = onehot_encoder.transform(df)
    ordinal_encoder.fit(df)

    feature_encoders = {
        "binary": binary_encoder,
        "onehot": onehot_encoder,
        "ordinal": ordinal_encoder,
    }

    return feature_encoders


def apply_encoders(df: pd.DataFrame, feature_encoders: dict) -> pd.DataFrame:
    """_summary_

    Args:
        feature_encoders (pickle.PickleDataSet): _description_

    Returns:
        pd.DataFrame: _description_
    """
    logger.info("Applying categorical encoders...")

    df = feature_encoders["binary"].transform(df)
    df = feature_encoders["onehot"].transform(df)
    df = feature_encoders["ordinal"].transform(df)

    print(df.shape)

    return df
