"""
Auxiliary functions for Kedro in version < 0.18 for loading pandas dataframes
in chunks.
"""
from typing import Iterator, Union

import pandas as pd
from kedro.io.core import get_filepath_str


def _load(self) -> pd.DataFrame:
    """Modified load function for CSVDataSet since in Kedro version < 0.18
    there is a bug with loading pandas dataset with chunksize parameter for
    bigger datasets.

    Returns:
        pd.DataFrame: loaded data frame
    """
    load_path = get_filepath_str(self._get_load_path(), self._protocol)

    return pd.read_csv(load_path, **self._load_args)


def _concat_chunks(chunks: Union[Iterator[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    """Auxiliary function for concatenating chunks into dataframe

    Args:
        chunks (Iterator[pd.DataFrame]): data chunks

    Returns:
        pd.DataFrame: dataframe
    """
    if isinstance(chunks, pd.DataFrame):
        df = chunks
    else:
        df = pd.DataFrame()
        for chunk in chunks:
            df = pd.concat([df, chunk], ignore_index=True)
    return df
