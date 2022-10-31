import pytest

from src.gid_ml_framework.pipelines.text_embeddings.nodes import (
    prepare_desciptions_and_labels,
)


def test_prepare_descriptions_correct_lengths(articles_dummy_df):
    result_descriptions, result_labels = prepare_desciptions_and_labels(
        articles_dummy_df
    )
    assert len(result_descriptions) == len(result_labels) == len(articles_dummy_df)


def test_prepare_descriptions_as_integers(articles_dummy_df):
    initial_df = articles_dummy_df
    initial_df["detail_desc"] = range(len(initial_df))
    result_descriptions, result_labels = prepare_desciptions_and_labels(initial_df)
    assert len(result_descriptions) == len(result_labels) == len(initial_df)


def test_prepare_descriptions_missing_column(articles_dummy_df):
    initial_df = articles_dummy_df.drop(["detail_desc"], axis=1)
    with pytest.raises(KeyError):
        prepare_desciptions_and_labels(initial_df)


def test_prepare_descriptions_correct_outputs(articles_dummy_df):
    result_descriptions, result_labels = prepare_desciptions_and_labels(
        articles_dummy_df
    )
    assert isinstance(result_descriptions, list) is True
    assert isinstance(result_labels, list) is True
    assert isinstance(result_descriptions[0], str) is True
    assert isinstance(result_labels[0], str) is True
