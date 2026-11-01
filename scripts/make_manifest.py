"""Write the manifest: one CSV row per labeled image, saying which split it belongs to.

Official val images become our test split. Official train images are split per class into
train and val. The official test images have no labels, so they are left out.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

FIELDS = ["image_id", "path", "wnid", "label", "split"]


def build_rows(raw_dir: Path, val_per_class: int, seed: int) -> list[dict[str, str | int]]:
    # wnids.txt is not sorted; sorting keeps each class's label independent of the file's order.
    wnids = sorted((raw_dir / "wnids.txt").read_text().split())
    labels = {wnid: label for label, wnid in enumerate(wnids)}
    rng = np.random.default_rng(seed)
    rows = []

    def add(path: Path, wnid: str, split: str) -> None:
        rows.append(
            {
                "image_id": path.stem,
                "path": path.as_posix(),
                "wnid": wnid,
                "label": labels[wnid],
                "split": split,
            }
        )

    for wnid in wnids:
        paths = sorted((raw_dir / "train" / wnid / "images").glob("*.JPEG"))
        val_indices = set(rng.permutation(len(paths))[:val_per_class].tolist())
        for i, path in enumerate(paths):
            add(path, wnid, "val" if i in val_indices else "train")

    val_dir = raw_dir / "val"
    annotations = (val_dir / "val_annotations.txt").read_text().splitlines()
    for line in sorted(annotations):
        filename, wnid = line.split("\t")[:2]
        add(val_dir / "images" / filename, wnid, "test")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/tiny-imagenet-200"))
    parser.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--val-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = build_rows(args.raw_dir, args.val_per_class, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
