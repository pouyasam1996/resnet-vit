# Data

[Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip): 200 classes of 64×64 color images. Everything here is versioned by DVC on S3; git only holds the pointers (`raw/tiny-imagenet-200.dvc`, `../dvc.lock`).

## Splits

| Split | Images | Per class | Source |
| --- | --- | --- | --- |
| `train` | 90,000 | 450 | official train |
| `val` | 10,000 | 50 | official train, held out for tuning |
| `test` | 10,000 | 50 | official val |

The official test images have no labels, so they are not used. The split is set by `split` in `../params.yaml`.

## Files

| File | Contents |
| --- | --- |
| `raw/tiny-imagenet-200/` | the extracted dataset |
| `manifest.csv` | one row per image: `image_id`, `path`, `wnid`, `label`, `split` |
| `processed/<split>_images.npy` | `uint8` array, `N × 64 × 64 × 3` (RGB) |
| `processed/<split>_labels.npy` | `int64` array, `N`, in the same order as the images |
| `processed/stats.json` | per-channel `mean` and `std` for images scaled to [0, 1], from `train` only |

Labels index the sorted class IDs in `raw/tiny-imagenet-200/wnids.txt`. Grayscale images are converted to RGB.

## Usage

```bash
uv run dvc pull                # download everything from S3 (needs AWS credentials)
uv run dvc repro               # rebuild whatever is out of date
uv run pytest tests/test_data.py
```

The data tests are skipped when the files are absent, as in CI, which holds no AWS credentials.
