from pathlib import Path

import pytest
import torch

DATA_OUTPUTS = Path(__file__).parent.parent / "data/processed/stats.json"


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="needs a CUDA GPU")
    skip_data = pytest.mark.skip(reason="needs the DVC data: run dvc pull or dvc repro")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
        if "data" in item.keywords and not DATA_OUTPUTS.exists():
            item.add_marker(skip_data)
