import torch


def test_gpu() -> None:
    torch._C._cuda_init()
    return 0
