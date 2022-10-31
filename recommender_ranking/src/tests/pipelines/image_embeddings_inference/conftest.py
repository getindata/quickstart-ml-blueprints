import pytest
import torch


def dummy_tensor(sample_size, embedding_size):
    torch.manual_seed(42)
    tensor = torch.rand([sample_size, embedding_size])
    return tensor


def dummy_file_names(a: int, n: int):
    dummy_tuple = tuple(f"abc_{a}_{i}.jpg" for i in range(n))
    return dummy_tuple


@pytest.fixture
def dummy_pytorch_preds():
    preds = list()
    for i in range(5):
        tensor = dummy_tensor(32, 16)
        file_names = dummy_file_names(i, 32)
        preds.append((tensor, file_names))
    return preds
