# Easy Diffusion Model

English | [简体中文](README_zh-CN.md)

A beginner-friendly PyTorch project for training a DDPM image generation model. It supports:

- Two model architectures: a convolutional `UNet` and a simplified `DiT`
- Four prediction targets: `epsilon`, `x0`, `v`, and `score`
- YAML configuration, command-line overrides, and checkpoint resumption


## 1. Project Structure

```text
.
├── configs/
│   └── default.yaml       # Default configuration for a single consumer GPU
├── models/
│   ├── unet.py            # UNet with two downsampling stages
│   └── dit.py             # patchify -> Transformer -> unpatchify
├── data.py                # ImageNet-10 data loading
├── diffusion.py           # DDPM formulas, four targets, and reverse sampling
├── train.py               # Straightforward training loop
└── sample.py              # Generate images from a checkpoint
```

## 2. Installation

[uv](https://docs.astral.sh/uv/) is recommended:

```bash
uv sync
```

You can also use an existing Python environment:

```bash
pip install -r requirements.txt
```

If you need a specific CUDA build, install `torch` and `torchvision` using the official PyTorch
installation command first, then install the remaining dependencies.

## 3. Dataset

The default configuration expects `datasets/imagenet-10` in the parent directory:

```text
../datasets/imagenet-10/
├── n02056570/
│   ├── xxx.JPEG
│   └── ...
├── n02085936/
└── ...
```

`torchvision.datasets.ImageFolder` discovers classes from the subdirectories. This project trains
an unconditional model, so the class labels are loaded but not passed to the network. Change
`dataset.path` in the YAML file if your dataset is stored elsewhere.

## 4. Small CPU Run

Without a GPU, run three optimization steps on real data with:

```bash
uv run python train.py --config configs/default.yaml \
  --set device=cpu \
  --set dataset.image_size=32 \
  --set dataset.batch_size=2 \
  --set dataset.num_workers=0 \
  --set model.unet.base_channels=32 \
  --set model.unet.time_embed_dim=128 \
  --set diffusion.timesteps=20 \
  --set training.epochs=1 \
  --set training.use_amp=false \
  --set training.sample_every_epochs=0 \
  --set training.max_steps=3
```

This command loads the dataset, adds forward-process noise, performs backpropagation, updates the
model, and saves a checkpoint. It skips the slower full sampling loop.

## 5. Training

Train the default UNet with epsilon prediction:

```bash
uv run python train.py --config configs/default.yaml
```

At startup, the script prints the complete configuration and these key fields:

```text
dataset, model, pred_type, image_size, batch_size,
timesteps, learning_rate, epochs, device
```

Select any of the four prediction targets:

```bash
uv run python train.py --config configs/default.yaml --set diffusion.pred_type=epsilon
uv run python train.py --config configs/default.yaml --set diffusion.pred_type=x0
uv run python train.py --config configs/default.yaml --set diffusion.pred_type=v
uv run python train.py --config configs/default.yaml --set diffusion.pred_type=score
```

Switch to DiT:

```bash
uv run python train.py --config configs/default.yaml --set model.name=dit
```

Multiple overrides can be combined. For example, run ten DiT optimization steps on a GPU:

```bash
uv run python train.py --config configs/default.yaml \
  --set model.name=dit \
  --set diffusion.pred_type=v \
  --set training.max_steps=10
```

To resume training, keep the model, image size, and diffusion settings identical to the checkpoint:

```bash
uv run python train.py --config configs/default.yaml \
  --resume runs/unet_epsilon_xxx/last.pt
```

## 6. Prediction Targets

The forward diffusion process has the following closed-form expression:

```text
x_t = a_t * x_0 + b_t * epsilon
a_t = sqrt(alpha_bar_t)
b_t = sqrt(1 - alpha_bar_t)
epsilon ~ N(0, I)
```

The network always receives `x_t: [B,3,H,W]` and `t: [B]`, and returns `[B,3,H,W]`. The meaning of
that output depends on `pred_type`:

| `pred_type` | Training target | Recovering `x0` from the output |
|---|---|---|
| `epsilon` | `epsilon` | `(x_t - b_t*epsilon) / a_t` |
| `x0` | `x_0` | The model output itself |
| `v` | `a_t*epsilon - b_t*x_0` | `a_t*x_t - b_t*v` |
| `score` | `-epsilon / b_t` | `(x_t + b_t²*score) / a_t` |

The `score` target contains `1/b_t`, which can become large at low noise levels. The implementation
therefore weights each image's score MSE by `b_t²`. This gives it the same scale as epsilon MSE
while the network still predicts the score directly.

## 7. Image Generation

```bash
uv run python sample.py \
  --checkpoint runs/unet_epsilon_xxx/last.pt \
  --num-images 16 \
  --output samples.png
```

Sampling starts from standard Gaussian noise with shape `[B,3,H,W]` and executes every configured
DDPM reverse step. The default 1,000-step process prioritizes a clear implementation over speed, so
it is slower than accelerated samplers such as DDIM.

## 8. Output Directory

Each training run creates a separate directory:

```text
runs/unet_epsilon_timestamp_config-hash/
├── config.yaml             # Complete resolved configuration
├── metrics.csv             # Loss history
├── last.pt                 # Latest checkpoint
├── epoch_XXXX.pt           # Periodic checkpoint
└── samples_epoch_XXXX.png  # Periodic sample grid
```

The default settings are a runnable teaching baseline, not a claim of optimal ImageNet-10
generation quality. Verify the complete pipeline first, then adjust `batch_size`, model width, and
training duration for your hardware.
