import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).parent.parent
EXPECTED_SIZES = {"train": 90_000, "val": 10_000, "test": 10_000}
EXPECTED_PER_CLASS = {"train": 450, "val": 50, "test": 50}

pytestmark = pytest.mark.data


def read_manifest() -> list[dict[str, str]]:
    with (ROOT / "data/manifest.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def test_manifest_split_sizes() -> None:
    rows = read_manifest()
    assert Counter(row["split"] for row in rows) == EXPECTED_SIZES
    assert len({row["path"] for row in rows}) == len(rows)

    per_class = Counter((row["split"], row["label"]) for row in rows)
    assert len(per_class) == 3 * 200
    for (split, _), count in per_class.items():
        assert count == EXPECTED_PER_CLASS[split]


def test_official_val_is_our_test_split() -> None:
    for row in read_manifest():
        assert ("/val/images/" in row["path"]) == (row["split"] == "test")


@pytest.mark.parametrize("split", EXPECTED_SIZES)
def test_arrays_match_manifest(split: str) -> None:
    images = np.load(ROOT / f"data/processed/{split}_images.npy", mmap_mode="r")
    labels = np.load(ROOT / f"data/processed/{split}_labels.npy")
    expected_labels = [int(row["label"]) for row in read_manifest() if row["split"] == split]

    assert images.shape == (EXPECTED_SIZES[split], 64, 64, 3)
    assert images.dtype == np.uint8
    assert labels.tolist() == expected_labels


def test_outputs_match_dvc_lock() -> None:
    lock = yaml.safe_load((ROOT / "dvc.lock").read_text())
    outs = [out for stage in lock["stages"].values() for out in stage["outs"]]
    assert len(outs) == 8
    for out in outs:
        assert md5(ROOT / out["path"]) == out["md5"], out["path"]
