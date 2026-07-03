# DEVLOG

## 1. 核心目标

训练一个无条件 DDPM，在 ImageNet-10 图像上学习生成分布。模型可选择卷积 UNet
或简化 DiT；监督参数化可选择 `epsilon`、`x0`、`v`、`score`。项目优先保证公式、
张量 shape 和训练数据流对初学者透明。

## 2. 算法与数据流

1. `ImageFolder` 读取 RGB 图像，缩放裁剪并归一化为 `x0: [B,3,H,W]`、范围 `[-1,1]`。
2. 均匀采样 `t: [B]` 与 `epsilon: [B,3,H,W]`，通过闭式公式得到
   `x_t = sqrt(alpha_bar_t)x0 + sqrt(1-alpha_bar_t)epsilon`。
3. UNet 或 DiT 接收 `x_t, t`，输出 `[B,3,H,W]`。
4. 依据 `pred_type` 构造 epsilon、x0、v 或 score 目标并计算 MSE；score-MSE 使用
   `sigma_t^2` 加权以稳定不同噪声水平的尺度。
5. 采样时把任意模型输出转换为 x0 预测，代入 DDPM 后验，从高斯噪声逐步生成图像。

## 3. 模块与实现计划

- `data.py`：数据增强与 DataLoader。
- `models/unet.py`、`models/dit.py`：两种时间条件模型。
- `diffusion.py`：前向加噪、四种目标、训练损失、DDPM 采样。
- `train.py`、`sample.py`：保持直接可读的训练与采样入口。
- `configs/`：单 GPU 默认训练配置。

## 4. 变更记录

| 日期 | 改了什么 | 为什么 | 涉及文件 |
|---|---|---|---|
| 2026-07-03 | 建立完整 DDPM 教学项目，加入两种模型、四种目标、配置、训练与采样 | 满足新手可读、直接运行、GPU/CPU 均可验证的需求 | 全部项目文件 |
| 2026-07-03 | 固定 CUDA 12.6 的 PyTorch 依赖并生成锁文件 | 避免 uv 把现有环境替换为不必要的 CUDA 13 大型依赖 | `pyproject.toml`、`requirements.txt`、`uv.lock` |
| 2026-07-03 | 完成四种目标、真实数据 CPU 训练与采样验证 | 确认公式、梯度、数据、checkpoint 和采样入口可用 | 本地验证产物（不发布） |
