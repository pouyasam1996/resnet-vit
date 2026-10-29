import pytest
import torch

import resnet_vit


def test_package_imports() -> None:
    assert resnet_vit.__name__ == "resnet_vit"


def test_tensor_sum():
    x = torch.ones(2, 3)
    assert x.sum().item() == 6.0


@pytest.mark.gpu
def test_cuda_tensor_sum():
    x = torch.ones(2, 3, device="cuda")
    assert x.sum().item() == 6.0
