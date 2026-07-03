# DEVLOG

## 1. 核心目标

在 ImageNet-10 图像上训练 DDPM。模型可选择无条件卷积 UNet，或以 ImageFolder
类别 ID 为条件的 DiT；监督参数化可选择 `epsilon`、`x0`、`v`、`score`。项目优先
保证公式、张量 shape 和训练数据流对初学者透明。

## 2. 算法与数据流

1. `ImageFolder` 读取 RGB 图像，缩放裁剪并归一化为 `x0: [B,3,H,W]`、范围 `[-1,1]`。
2. 均匀采样 `t: [B]` 与 `epsilon: [B,3,H,W]`，通过闭式公式得到
   `x_t = sqrt(alpha_bar_t)x0 + sqrt(1-alpha_bar_t)epsilon`。
3. UNet 接收 `x_t, t`；DiT 额外接收类别 `y: [B]`，计算
   `c = time_embedding(t) + class_embedding(y)`，并用 AdaLN-Zero 将 `c` 注入每个
   attention/MLP 子层。
4. 训练 DiT 时以可配置概率把 `y` 替换为空类别，使同一个模型同时学习条件与无条件
   预测；UNet 仍保持无条件。
5. 依据 `pred_type` 构造 epsilon、x0、v 或 score 目标并计算 MSE；score-MSE 使用
   `sigma_t^2` 加权以稳定不同噪声水平的尺度。
6. DiT 采样分别计算条件预测 `p_c` 与无条件预测 `p_u`，通过
   `p = p_u + cfg_scale * (p_c - p_u)` 做 classifier-free guidance，再把输出转换为
   x0 预测并执行 DDPM 后验采样。

## 3. 模块与实现计划

- `data.py`：数据增强与 DataLoader。
- `models/unet.py`、`models/dit.py`：无条件 UNet 与类别条件 AdaLN-Zero DiT。
- `diffusion.py`：前向加噪、四种目标、训练损失、DDPM 采样。
- `train.py`、`sample.py`：标签训练、类别选择与 CFG 采样入口。
- `configs/`：单 GPU 默认训练配置。

## 4. 变更记录

| 日期 | 改了什么 | 为什么 | 涉及文件 |
|---|---|---|---|
| 2026-07-03 | 建立完整 DDPM 教学项目，加入两种模型、四种目标、配置、训练与采样 | 满足新手可读、直接运行、GPU/CPU 均可验证的需求 | 全部项目文件 |
| 2026-07-03 | 固定 CUDA 12.6 的 PyTorch 依赖并生成锁文件 | 避免 uv 把现有环境替换为不必要的 CUDA 13 大型依赖 | `pyproject.toml`、`requirements.txt`、`uv.lock` |
| 2026-07-03 | 完成四种目标、真实数据 CPU 训练与采样验证 | 确认公式、梯度、数据、checkpoint 和采样入口可用 | 本地验证产物（不发布） |
| 2026-07-03 | 确定类别条件 DiT 与 CFG 的数据流和实现计划 | 用现有 ImageFolder 类别监督 AdaLN-Zero，并通过标签丢弃学习无条件分支 | `DEVLOG.md` |
| 2026-07-03 | 实现类别条件 AdaLN-Zero DiT、标签丢弃、类别映射和 CFG 采样接口 | 让 ImageFolder 类别可控制生成内容，同时保留无条件 UNet | `models/dit.py`、`models/__init__.py`、`diffusion.py`、`train.py`、`sample.py`、`data.py`、`utils.py`、`configs/default.yaml`、`checks/`、README |
| 2026-07-03 | 完成 18 项测试及真实数据最小训练、断点恢复、指定类别 CFG 采样验证 | 确认类别条件端到端链路、旧 UNet 路径和 checkpoint 元数据均可用 | `checks/`、本地验证产物（不发布） |
