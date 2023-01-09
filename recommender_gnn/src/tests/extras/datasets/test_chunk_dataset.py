import pandas as pd

from recommender_gnn.extras.datasets.chunks_dataset import _concat_chunks


def test_concat_chunks_given_chunks_should_return_dataframe(custom_chunks):
    df = _concat_chunks(custom_chunks)
    assert isinstance(df, pd.DataFrame)


def test_concat_chunks_given_dataframe_should_return_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    concat_df = _concat_chunks(df)
    assert isinstance(concat_df, pd.DataFrame)
    assert concat_df.equals(df)
