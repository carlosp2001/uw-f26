# CS680 Assignment 2

This directory contains the notebook for Assignment 2 (VGG-11 on MNIST, baseline vs. augmented):

- `a2.ipynb`
- `vgg11_mnist_baseline.pt`, `vgg11_mnist_augmented.pt` (saved model weights)
- `data/MNIST/` (downloaded automatically by the notebook)

## Machine

- MacBook with Apple Silicon (M1, `arm64`), macOS
- Training runs on the Apple GPU through PyTorch's **MPS** backend (the notebook picks CUDA, then MPS, then CPU)

## Environment

A dedicated conda environment named `uw-f26` was used, on top of Anaconda3 (`/opt/anaconda3`).

Versions printed by the notebook when it was run:

| Package       | Version |
|---------------|---------|
| Python        | 3.12.12 |
| PyTorch       | 2.5.1   |
| torchvision   | 0.20.1  |
| Matplotlib    | 3.11.1  |
| NumPy         | 2.5.2   |

Other packages installed in the environment for running the notebook:

- `notebook` 7.6.3 (installed with pip)
- `ipykernel` 7.3.0

PyTorch and torchvision were installed from the `pytorch` conda channel (the `osx-arm64` builds, which include MPS support).

## Setup

```bash
conda create -n uw-f26 python=3.12
conda activate uw-f26

conda install -c pytorch pytorch=2.5.1 torchvision=0.20.1
conda install -c conda-forge numpy matplotlib ipykernel
pip install notebook
```

## Usage

```bash
conda activate uw-f26
cd CS680/assignments/a2
jupyter notebook a2.ipynb
```

Then select **Run All**. The first run downloads MNIST into `./data`. The notebook prints `Device: mps` when the Apple GPU is in use.

## Notes

- The seed is fixed (`SEED = 480`), but MPS kernels are not guaranteed to be bit-for-bit deterministic, so small numeric differences between runs are possible.
- The `uw-f26` environment also has `torch` 2.14.1 and `torchvision` 0.29.1 installed with pip, next to the conda builds. In that environment `import torch` loads 2.5.1, which is the version the results were produced with.
