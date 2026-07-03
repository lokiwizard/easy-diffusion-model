# Easy Diffusion Model

[English](README.md) | 简体中文

这是一个面向扩散模型初学者的 PyTorch DDPM 项目。它支持：

- 两种模型：无条件卷积 `UNet`、类别条件 AdaLN-Zero `DiT`
- 四种预测目标：`epsilon`、`x0`、`v`、`score`
- DiT 的 classifier-free guidance（CFG）采样
- YAML 配置、命令行单项覆盖、断点续训


## 1. 项目结构

```text
.
├── configs/
│   └── default.yaml       # 单张普通 GPU 的默认配置
├── models/
│   ├── unet.py            # 两层下采样 UNet
│   └── dit.py             # patchify -> Transformer -> unpatchify
├── data.py                # ImageNet-10 数据读取
├── diffusion.py           # DDPM 公式、四种目标、反向采样
├── train.py               # 直接可读的训练循环
└── sample.py              # 从 checkpoint 生成图像
```

## 2. 安装

推荐使用 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync
```

也可以使用已有 Python 环境：

```bash
pip install -r requirements.txt
```

若需要特定 CUDA 版本的 PyTorch，请优先按 PyTorch 官方安装命令安装 `torch` 和
`torchvision`，再安装其余依赖。

## 3. 数据集

默认配置使用项目上一级目录中的 `datasets/imagenet-10`：

```text
../datasets/imagenet-10/
├── n02056570/
│   ├── xxx.JPEG
│   └── ...
├── n02085936/
└── ...
```

`torchvision.datasets.ImageFolder` 会按类别目录名排序并生成类别 ID。选择 DiT 时，这些
类别 ID 会作为生成条件；选择 UNet 时仍训练无条件模型。路径不同可修改 YAML 的
`dataset.path`。

## 4. CPU 小规模运行

没有 GPU 时，可用真实数据只跑 3 个优化步骤：

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

该命令会读取数据、前向加噪、反传、更新参数并保存 checkpoint，但默认不执行较慢的
完整采样。

## 5. 训练

默认 UNet + epsilon 预测：

```bash
uv run python train.py --config configs/default.yaml
```

训练开始时会打印完整配置和以下关键项：

```text
dataset, model, pred_type, image_size, batch_size,
timesteps, learning_rate, epochs, device
```

切换四种预测目标：

```bash
uv run python train.py --config configs/default.yaml --set diffusion.pred_type=epsilon
uv run python train.py --config configs/default.yaml --set diffusion.pred_type=x0
uv run python train.py --config configs/default.yaml --set diffusion.pred_type=v
uv run python train.py --config configs/default.yaml --set diffusion.pred_type=score
```

切换到 DiT：

```bash
uv run python train.py --config configs/default.yaml --set model.name=dit
```

DiT 使用 `time_embedding + class_embedding` 驱动每层 AdaLN-Zero。配置中的
`model.dit.num_classes` 必须与 ImageFolder 的类别目录数一致；训练时
`model.dit.class_dropout_prob` 比例的标签会被替换为空类别，使模型学会 CFG 所需的
无条件分支。类别 ID 与目录名的映射会在训练开始时打印并保存在 checkpoint 中。

多个参数可连续覆盖。例如先用 GPU 做 10 step 快速检查：

```bash
uv run python train.py --config configs/default.yaml \
  --set model.name=dit \
  --set diffusion.pred_type=v \
  --set training.max_steps=10
```

断点续训时应使用相同的模型、图像尺寸和扩散配置：

```bash
uv run python train.py --config configs/default.yaml \
  --resume runs/unet_epsilon_xxx/last.pt
```

## 6. 四种预测目标

前向加噪的闭式公式是：

```text
x_t = a_t * x_0 + b_t * epsilon
a_t = sqrt(alpha_bar_t)
b_t = sqrt(1 - alpha_bar_t)
epsilon ~ N(0, I)
```

网络始终输入 `x_t: [B,3,H,W]` 和 `t: [B]`；DiT 还输入 `y: [B]` 类别 ID。输出
都是 `[B,3,H,W]`，但输出含义由 `pred_type` 决定：

| `pred_type` | 训练目标 | 从输出恢复 `x0` |
|---|---|---|
| `epsilon` | `epsilon` | `(x_t - b_t*epsilon) / a_t` |
| `x0` | `x_0` | 模型输出本身 |
| `v` | `a_t*epsilon - b_t*x_0` | `a_t*x_t - b_t*v` |
| `score` | `-epsilon / b_t` | `(x_t + b_t²*score) / a_t` |

`score` 目标在小噪声时含有 `1/b_t`，数值可能很大。因此代码对每张图的 score-MSE
乘 `b_t²`；这与 epsilon-MSE 的尺度等价，同时网络仍然直接输出 score。

## 7. 生成图像

```bash
uv run python sample.py \
  --checkpoint runs/unet_epsilon_xxx/last.pt \
  --num-images 16 \
  --output samples.png
```

类别条件 DiT 可使用类别 ID 或类别目录名：

```bash
uv run python sample.py \
  --checkpoint runs/dit_epsilon_xxx/last.pt \
  --num-images 8 \
  --class-labels n02085936 \
  --cfg-scale 4.0 \
  --output class_samples.png
```

单个类别会应用到全部图像；也可传入与图像数相同的逗号分隔类别，例如
`--class-labels 0,0,1,1`。省略该参数时会依次循环所有类别。CFG 使用
`uncond + cfg_scale * (cond - uncond)`；`cfg_scale=1.0` 表示仅使用条件预测。

采样从 `[B,3,H,W]` 标准高斯噪声开始，执行配置中的全部 DDPM 时间步。默认 1000 步
强调公式清晰而非采样速度，因此生成会比 DDIM 等加速采样器慢。

## 8. 输出目录

每次新训练创建独立目录：

```text
runs/unet_epsilon_时间戳_配置哈希/
├── config.yaml             # 本次实际使用的完整配置
├── metrics.csv             # loss 记录
├── last.pt                 # 最近 checkpoint
├── epoch_XXXX.pt           # 周期 checkpoint
└── samples_epoch_XXXX.png  # 周期采样网格
```

默认参数是可运行的教学基线，不代表针对 ImageNet-10 的最佳生成质量。先确认流程正确，
再根据显存调整 `batch_size`、模型宽度和训练轮数。
