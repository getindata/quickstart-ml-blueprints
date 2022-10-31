import pandas as pd

from src.gid_ml_framework.pipelines.image_embeddings_inference.nodes import (
    _stack_predictions,
)


def test_stack_predictions_check_sizes(dummy_pytorch_preds):
    embedding_size = 16
    results_df = _stack_predictions(dummy_pytorch_preds, embedding_size)
    assert len(results_df.columns) == embedding_size
    assert sum(1 for col in results_df.columns if col.startswith("img_emb_")) == 16
    assert len(results_df) == 32 * 5


def test_stack_predictions_check_single_batch(dummy_pytorch_preds):
    embedding_size = 16
    # indexing [:1] instead of [0], so it returns list, not a tuple
    predictions = dummy_pytorch_preds[:1]
    results_df = _stack_predictions(predictions, embedding_size)
    assert len(results_df.columns) == embedding_size
    assert sum(1 for col in results_df.columns if col.startswith("img_emb_")) == 16
    assert len(results_df) == 32


def test_stack_predictions_string_emb_size(dummy_pytorch_preds):
    embedding_size = "16"
    results_df = _stack_predictions(dummy_pytorch_preds, embedding_size)
    assert len(results_df.columns) == 16
    assert sum(1 for col in results_df.columns if col.startswith("img_emb_")) == 16
    assert len(results_df) == 32 * 5


def test_stack_predictions_missing_extension_in_filename(dummy_pytorch_preds):
    # remove extension from filename
    predictions = dummy_pytorch_preds[:1]
    unzipped = zip(*predictions)
    tensors, filenames = unzipped
    tensors = tensors[0]
    filenames_without_extension = tuple(file.split(".")[0] for file in filenames[0])
    new_predictions = list()
    new_predictions.append((tensors, filenames_without_extension))
    # compare files with and without extension
    embedding_size = 16
    results_with_df = _stack_predictions(predictions, embedding_size)
    results_without_df = _stack_predictions(new_predictions, embedding_size)
    pd.testing.assert_frame_equal(results_with_df, results_without_df)
    assert len(results_without_df) == len(results_with_df)
