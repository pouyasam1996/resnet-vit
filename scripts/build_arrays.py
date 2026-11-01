"""Turn the manifest's images into one array file per split, plus normalization stats.

Writes <split>_images.npy (uint8, N x 64 x 64 x 3), <split>_labels.npy (int64, N) and stats.json.
Mean and std are computed from the train split only, so val and test cannot leak into them.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SHAPE = (64, 64, 3)
SPLITS = ("train", "val", "test")
BATCH_SIZE = 4096


def write_split(rows: list[dict[str, str]], out_dir: Path, split: str) -> None:
    images = np.lib.format.open_memmap(
        out_dir / f"{split}_images.npy", mode="w+", dtype=np.uint8, shape=(len(rows), *IMAGE_SHAPE)
    )
    for i, row in enumerate(rows):
        with Image.open(row["path"]) as image:
            # A few images are grayscale; RGB gives every image the same shape.
            images[i] = np.asarray(image.convert("RGB"), dtype=np.uint8)
    images.flush()
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    np.save(out_dir / f"{split}_labels.npy", labels)


def channel_stats(images: np.ndarray) -> tuple[list[float], list[float]]:
    """Per-channel mean and std of images scaled to [0, 1]."""
    total = np.zeros(3)
    total_squared = np.zeros(3)
    for start in range(0, len(images), BATCH_SIZE):
        batch = images[start : start + BATCH_SIZE].astype(np.float64) / 255
        total += batch.sum(axis=(0, 1, 2))
        total_squared += (batch**2).sum(axis=(0, 1, 2))
    n_pixels = len(images) * IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    mean = total / n_pixels
    std = np.sqrt(total_squared / n_pixels - mean**2)
    return mean.tolist(), std.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    with args.manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        write_split([row for row in rows if row["split"] == split], args.out_dir, split)

    train_images = np.load(args.out_dir / "train_images.npy", mmap_mode="r")
    mean, std = channel_stats(train_images)
    stats = {"mean": mean, "std": std}
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
