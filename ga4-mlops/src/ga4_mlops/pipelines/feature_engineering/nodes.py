"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.4
"""
import logging

import category_encoders as ce
import pandas as pd
from sklearn.impute import SimpleImputer

from ..data_preparation_utils import (
    clean_column_names,
    ensure_column_types,
    extract_column_names,
)

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply manual feature engineering transformations.

    Args:
        df (pd.DataFrame): data frame with raw features

    Returns:
        pd.DataFrame: data frame after feature engineering
    """
    logger.info("Applying manual feature engineering transformations...")

    df["c_weekday"] = pd.to_datetime(df["i_visit_start_time"], unit="us").dt.weekday
    df["c_visit_start_hour"] = pd.to_datetime(
        df["i_visit_start_time"], unit="us"
    ).dt.hour

    return df


def exclude_features(df: pd.DataFrame, features_to_exclude: list) -> pd.DataFrame:
    """Exclude manually selected features.

    Args:
        df (pd.DataFrame): data frame after manual feature engineering
        features_to_exclude (list): list of features to exclude

    Returns:
        pd.DataFrame: data frame with selected features excluded
    """
    logger.info(f"Excluding selected features: {features_to_exclude}")

    preserved_features = [col for col in df.columns if col not in features_to_exclude]
    df = df.loc[:, preserved_features]

    return df


def fit_imputers(df: pd.DataFrame, imputation_strategies: dict) -> dict:
    """Fit imputers for missing values using selected strategies.

    Args:
        df (pd.DataFrame): data frame after feature exclusion
        imputation_strategies (dict): dictionary of imputation strategies for different features

    Returns:
        dict: dictionary of imputer objects
    """
    logger.info("Fitting missing values imputers...")

    columns_to_impute = [
        item for sublist in list(imputation_strategies.values()) for item in sublist
    ]
    assert all(
        item in df.columns for item in columns_to_impute
    ), "Some of columns to encode are not in df"

    _, num_cols, cat_cols, _ = extract_column_names(df)
    df = ensure_column_types(df, num_cols, cat_cols)

    num_mean_imputer = _imputer_fit(df, imputation_strategies, "mean")
    num_zero_imputer = _imputer_fit(df, imputation_strategies, "zero")
    cat_mostfreq_imputer = _imputer_fit(df, imputation_strategies, "mostfreq")
    cat_unknown_imputer = _imputer_fit(df, imputation_strategies, "unknown")

    imputers = {
        "mean": num_mean_imputer,
        "zero": num_zero_imputer,
        "mostfreq": cat_mostfreq_imputer,
        "unknown": cat_unknown_imputer,
    }

    return imputers


def apply_imputers(df: pd.DataFrame, imputers: dict) -> pd.DataFrame:
    """Apply fitted imputers on a data frame.

    Args:
        df (pd.DataFrame): data frame after feature exclusion
        imputers (dict): dictionary of imputer objects

    Returns:
        pd.DataFrame: imputed data frame
    """
    logger.info("Applying missing values imputers...")

    _, num_cols, cat_cols, _ = extract_column_names(df)
    df = ensure_column_types(df, num_cols, cat_cols)

    df = _imputer_transform(df, imputers, "mean")
    df = _imputer_transform(df, imputers, "zero")
    df = _imputer_transform(df, imputers, "mostfreq")
    df = _imputer_transform(df, imputers, "unknown")

    return df


def fit_encoders(df: pd.DataFrame, encoder_types: dict) -> dict:
    """Fit categorical encoders using selected types.

    Args:
        df (pd.DataFrame): data frame after imputation
        encoder_types (dict): dictionary of encoder types for different features
    Returns:
        pd.DataFrame: dictionary of encoder objects
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
    """Apply fitted encoders on a data frame.

    Args:
        df (pd.DataFrame): _description_
        feature_encoders (dict): dictionary of encoder objects

    Returns:
        pd.DataFrame: encoded data frame
    """
    logger.info("Applying categorical encoders...")

    df = feature_encoders["binary"].transform(df)
    df = feature_encoders["onehot"].transform(df)
    df = feature_encoders["ordinal"].transform(df)

    df = clean_column_names(df)

    return df


def _imputer_fit(
    df: pd.DataFrame, imputation_strategies: dict, selected_strategy: str
) -> SimpleImputer:
    """Fit imputers on a given data frame using selected imputation strategy.

    Args:
        df (pd.DataFrame): data frame as a base for imputer fit
        selected_strategy (str): selected imputation strategy

    Returns:
        SimpleImputer: fitted imputer object
    """
    if selected_strategy == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif selected_strategy == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    elif selected_strategy == "mostfreq":
        imputer = SimpleImputer(strategy="most_frequent")
    elif selected_strategy == "unknown":
        imputer = SimpleImputer(strategy="constant", fill_value="UNKNOWN")
    else:
        imputer = None

    if len(imputation_strategies[selected_strategy]) > 0:
        imputer.fit(df[imputation_strategies[selected_strategy]])
    else:
        imputer = None

    return imputer


def _imputer_transform(
    df: pd.DataFrame, imputers: dict, selected_strategy: str
) -> pd.DataFrame:
    """Apply imputer with a selected strategy, if exists.

    Args:
        df (pd.DataFrame): data frame to impute
        imputers (dict): imputers dictionary
        selected_strategy (str): selected imputation strategy from dictionary

    Returns:
        pd.DataFrame: imputed data frame
    """
    if imputers[selected_strategy] is not None:
        df[imputers[selected_strategy].feature_names_in_] = imputers[
            selected_strategy
        ].transform(df[imputers[selected_strategy].feature_names_in_])

    return df
