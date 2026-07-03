"""DDPM 的前向加噪、训练目标转换和反向采样。

符号约定：
    beta_t                  : 第 t 步加入的噪声方差
    alpha_t = 1 - beta_t
    alpha_bar_t             : alpha_0 * alpha_1 * ... * alpha_t
    sigma_t = sqrt(1 - alpha_bar_t)

所有时间相关系数保存为 shape [T] 的 buffer，会跟随模块自动移动到 GPU/CPU，
但不会被优化器更新。
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm


VALID_PREDICTION_TYPES = {"epsilon", "x0", "v", "score"}


def _extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: Sequence[int]) -> torch.Tensor:
    """从 [T] 系数表中取出一个 batch 的系数，并扩展到可与图像广播的 shape。

    values: [T]
    timesteps: [B]
    返回值: [B,1,1,1]（当 x_shape 是四维图像 shape 时）
    """
    batch_values = values.gather(0, timesteps)  # [B]
    return batch_values.reshape(timesteps.shape[0], *([1] * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """固定线性 beta 调度的 DDPM。

    该类没有可训练参数。神经网络只负责 model(x_t, t)，扩散公式集中在这里，
    便于对照公式理解四种预测目标之间的关系。
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        pred_type: str = "epsilon",
    ) -> None:
        super().__init__()
        if pred_type not in VALID_PREDICTION_TYPES:
            raise ValueError(
                f"pred_type={pred_type!r} 无效，可选值：{sorted(VALID_PREDICTION_TYPES)}"
            )
        if timesteps < 2:
            raise ValueError("timesteps 至少为 2")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("需要满足 0 < beta_start < beta_end < 1")

        self.timesteps = timesteps
        self.pred_type = pred_type

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)  # [T]
        alphas = 1.0 - betas  # [T]
        alpha_bars = torch.cumprod(alphas, dim=0)  # [T]
        alpha_bars_previous = F.pad(alpha_bars[:-1], (1, 0), value=1.0)  # [T]

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_previous", alpha_bars_previous)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

        # q(x_{t-1} | x_t, x_0) 的闭式高斯后验：
        # mean = coef1 * x_0 + coef2 * x_t
        posterior_variance = betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
        posterior_mean_coef1 = (
            betas * alpha_bars_previous.sqrt() / (1.0 - alpha_bars)
        )
        posterior_mean_coef2 = (
            (1.0 - alpha_bars_previous) * alphas.sqrt() / (1.0 - alpha_bars)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def q_sample(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """直接采样任意时间步 x_t，而不必逐步加噪。

        DDPM 前向过程的重参数化公式：

            x_t = sqrt(alpha_bar_t) * x_0
                  + sqrt(1 - alpha_bar_t) * epsilon,
            epsilon ~ N(0, I).

        输入 clean_images/x_0: [B,3,H,W]，timesteps: [B]。
        返回 noisy_images/x_t 和 noise/epsilon，二者均为 [B,3,H,W]。
        """
        if noise is None:
            noise = torch.randn_like(clean_images)  # [B,3,H,W]
        sqrt_alpha_bar = _extract(self.sqrt_alpha_bars, timesteps, clean_images.shape)
        sigma = _extract(self.sqrt_one_minus_alpha_bars, timesteps, clean_images.shape)
        noisy_images = sqrt_alpha_bar * clean_images + sigma * noise  # [B,3,H,W]
        return noisy_images, noise

    def training_target(
        self,
        clean_images: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """根据 pred_type 构造监督目标，返回 shape [B,3,H,W]。

        四种参数化都描述同一个去噪问题：

        epsilon: 目标是前向过程中加入的标准高斯噪声 epsilon。

        x0:      目标是原始干净图像 x_0。

        v:       v = sqrt(alpha_bar_t) * epsilon
                       - sqrt(1-alpha_bar_t) * x_0。
                 v 参数化在高噪声和低噪声区域之间通常更均衡。

        score:   条件高斯 q(x_t|x_0) 对 x_t 的 score（对数密度梯度）：
                 score = grad_x_t log q(x_t|x_0)
                       = -epsilon / sqrt(1-alpha_bar_t)。
        """
        sqrt_alpha_bar = _extract(self.sqrt_alpha_bars, timesteps, clean_images.shape)
        sigma = _extract(self.sqrt_one_minus_alpha_bars, timesteps, clean_images.shape)

        if self.pred_type == "epsilon":
            return noise
        if self.pred_type == "x0":
            return clean_images
        if self.pred_type == "v":
            return sqrt_alpha_bar * noise - sigma * clean_images
        return -noise / sigma.clamp(min=1e-8)  # score: [B,3,H,W]

    def training_loss(
        self,
        model: nn.Module,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """计算一个 batch 的简化 DDPM 均方误差损失。"""
        noisy_images, noise = self.q_sample(clean_images, timesteps)  # 各 [B,3,H,W]
        target = self.training_target(clean_images, noise, timesteps)  # [B,3,H,W]
        prediction = model(noisy_images, timesteps)  # [B,3,H,W]
        if prediction.shape != target.shape:
            raise RuntimeError(
                f"模型输出 shape {tuple(prediction.shape)} 与目标 {tuple(target.shape)} 不同"
            )

        # 先对 C,H,W 求均值，保留每张图一个 loss：[B,3,H,W] -> [B]。
        per_image_loss = F.mse_loss(prediction, target, reduction="none").flatten(1).mean(1)
        if self.pred_type == "score":
            # score 的幅值含 1/sigma_t；若直接 MSE，接近 t=0 时会异常大。
            # 乘 sigma_t^2 后，优化尺度与 epsilon-MSE 等价，但网络输出仍是 score。
            sigma_squared = _extract(
                1.0 - self.alpha_bars, timesteps, clean_images.shape
            ).flatten()  # [B]
            per_image_loss = per_image_loss * sigma_squared
        return per_image_loss.mean()  # 标量

    def model_output_to_x0(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """把任意预测参数化转换为 x_0 预测，输入输出均为 [B,3,H,W]。

        令 a=sqrt(alpha_bar_t)，b=sqrt(1-alpha_bar_t)，且 x_t=a*x_0+b*epsilon。

        epsilon 参数化：x_0 = (x_t - b*epsilon) / a
        x0 参数化：     x_0 = model_output
        v 参数化：      x_0 = a*x_t - b*v
        score 参数化：  epsilon = -b*score，
                        x_0 = (x_t + b^2*score) / a
        """
        a = _extract(self.sqrt_alpha_bars, timesteps, noisy_images.shape)
        b = _extract(self.sqrt_one_minus_alpha_bars, timesteps, noisy_images.shape)

        if self.pred_type == "epsilon":
            return (noisy_images - b * model_output) / a.clamp(min=1e-8)
        if self.pred_type == "x0":
            return model_output
        if self.pred_type == "v":
            return a * noisy_images - b * model_output
        return (noisy_images + b.square() * model_output) / a.clamp(min=1e-8)

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """执行一步 x_t -> x_{t-1}，输入输出均为 [B,3,H,W]。"""
        model_output = model(noisy_images, timesteps)  # [B,3,H,W]
        predicted_x0 = self.model_output_to_x0(
            noisy_images, timesteps, model_output
        ).clamp(-1.0, 1.0)  # [B,3,H,W]

        coef1 = _extract(self.posterior_mean_coef1, timesteps, noisy_images.shape)
        coef2 = _extract(self.posterior_mean_coef2, timesteps, noisy_images.shape)
        posterior_mean = coef1 * predicted_x0 + coef2 * noisy_images  # [B,3,H,W]

        variance = _extract(self.posterior_variance, timesteps, noisy_images.shape)
        random_noise = torch.randn_like(noisy_images)  # [B,3,H,W]
        # t=0 时已经得到 x_0，不应再加噪声。mask: [B,1,1,1]。
        nonzero_time_mask = (timesteps != 0).float().reshape(
            timesteps.shape[0], *([1] * (noisy_images.ndim - 1))
        )
        return posterior_mean + nonzero_time_mask * variance.clamp(min=1e-20).sqrt() * random_noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        device: torch.device,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """从 x_T~N(0,I) 开始逐步采样，最终返回 [B,3,H,W]、范围约 [-1,1]。"""
        was_training = model.training
        model.eval()
        images = torch.randn(shape, device=device)  # x_T: [B,3,H,W]
        reverse_steps = range(self.timesteps - 1, -1, -1)
        if show_progress:
            reverse_steps = tqdm(reverse_steps, desc="DDPM sampling", leave=False)

        for step in reverse_steps:
            timesteps = torch.full(
                (shape[0],), step, device=device, dtype=torch.long
            )  # [B]
            images = self.p_sample(model, images, timesteps)  # [B,3,H,W]

        if was_training:
            model.train()
        return images.clamp(-1.0, 1.0)
