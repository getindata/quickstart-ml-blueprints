"""This module contains common modeling functionalities for different pipelines."""
from typing import Any

import numpy as np
import pandas as pd

from .data_preparation_utils import extract_column_names


def score_abt(abt: pd.DataFrame, model: Any) -> np.ndarray:
    """Calculate predicted scores on q given ABT.

    Args:
        abt (pd.DataFrame): ABT to score
        model (Any): any model with `predict_proba` method

    Returns:
        np.ndarray: predicted scores
    """
    _, num_cols, cat_cols, _ = extract_column_names(abt)

    scores = model.predict_proba(abt[num_cols + cat_cols])[:, 1]

    return scores
