# resnet-vit

[![CI](https://github.com/pouyasam1996/resnet-vit/actions/workflows/ci.yml/badge.svg)](https://github.com/pouyasam1996/resnet-vit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

A ResNet and a Vision Transformer (ViT), written from scratch in PyTorch and trained on Tiny ImageNet, to measure **which training choices actually matter and why**, on one and two GPUs, then deployed with ONNX and TensorRT.

The project is built the way a production ML team works: pinned environments, automated checks on every change, versioned data, tracked experiments, containerized training and verified deployment.

> **Status:** Stage 0 (repository foundation) is complete. Results will appear here as stages finish.

## Questions this project answers

1. How do the activation function, normalization, learning-rate schedule and augmentation each affect accuracy, measured against run-to-run noise?
2. Why does a small ViT lose to a ResNet at this data size, and how much does heavy augmentation close the gap?
3. How close to 2× does training get on two GPUs, and where does the rest go?
4. How much accuracy do FP16 and INT8 TensorRT engines give up for their speedup?

## Roadmap

| Stage | Scope | Status |
| --- | --- | --- |
| 0 | Repo foundation: uv, ruff, pre-commit, pytest, GitHub Actions CI, protected `main` | Done (`v0.0`) |
| 1 | Data: Tiny ImageNet splits and preprocessing, versioned with DVC on S3 | Planned |
| 2 | Models from scratch: ResNet-18 adapted to 64×64 images, small ViT | Planned |
| 3 | Single-GPU training: Hydra configs, mixed precision, checkpoint/resume, EMA, experiment tracking | Planned |
| 4 | Multi-GPU training with DDP | Planned |
| 5 | Ablation study | Planned |
| 6 | Docker image for running training on any rented GPU | Planned |
| 7 | Deployment: ONNX export, TensorRT FP16/INT8, accuracy parity and latency benchmarks | Planned |
| 8 | Write-up: results, what mattered and why, what failed and why | Planned |

## Tech stack

| Area | Tools |
| --- | --- |
| Modeling and training | PyTorch, torchvision |
| Environment | uv (exact versions pinned in `uv.lock`) |
| Code quality | ruff (lint + format), pre-commit, pytest |
| CI | GitHub Actions: lint, format check and tests on every pull request |
| Data versioning | DVC with an S3 remote (Stage 1) |
| Configs and experiments | Hydra, Weights & Biases (Stage 3) |
| Distributed training | torchrun + DDP (Stage 4) |
| Packaging | Docker (Stage 6) |
| Deployment | ONNX, ONNX Runtime, TensorRT (Stage 7) |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and git.

```bash
git clone https://github.com/pouyasam1996/resnet-vit.git
cd resnet-vit
uv sync                       # installs Python 3.11 and the exact locked dependencies
uv run pre-commit install     # enables the local commit checks
uv run pytest                 # runs the tests; GPU tests are skipped without CUDA
```

## Repository layout

```text
resnet-vit/
├── .github/workflows/ci.yml   # CI: lint, format check, tests
├── src/resnet_vit/            # the package (models, data, training, deployment as stages land)
├── tests/                     # pytest; conftest.py skips GPU tests on machines without CUDA
├── .pre-commit-config.yaml    # local checks run before every commit
├── pyproject.toml             # dependencies and tool settings
└── uv.lock                    # exact pinned versions
```

## Development workflow

- `main` is protected: every change arrives through a pull request, and CI must pass before merging.
- One branch per stage. Each completed stage is tagged (`v0.0`, `v0.1`, …), so every result can be traced to exact code.
- Local checks (pre-commit) catch problems early; CI is the final gate.
- Tests that need a GPU are marked `gpu` and run on GPU machines only.

| Command | What it does |
| --- | --- |
| `uv run ruff check . --fix` | Lint and auto-fix |
| `uv run ruff format .` | Format |
| `uv run pytest -v` | Run tests |
| `uv run pre-commit run --all-files` | Run every local check on the whole repo |

## Results

Coming with Stage 5 (ablations) and Stage 7 (deployment).

## License

[MIT](LICENSE)
