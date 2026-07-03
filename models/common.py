"""UNet 和 DiT 共用的时间步编码。"""

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """把整数扩散时间步编码成连续向量。

    公式与 Transformer 的位置编码相同。对第 i 对通道：

        embedding(t, 2i)     = sin(t / 10000^(2i/D))
        embedding(t, 2i + 1) = cos(t / 10000^(2i/D))

    不同频率让模型既能区分相邻时间步，也能感知较大的时间尺度。
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 4:
            raise ValueError("时间编码维度至少应为 4")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """参数 timesteps 的 shape 为 [B]，返回 shape 为 [B, D]。"""
        half_dim = self.embedding_dim // 2

        # frequencies: [D/2]。这些频率是常量，无须作为可训练参数。
        exponent = -math.log(10_000) * torch.arange(
            half_dim, device=timesteps.device, dtype=torch.float32
        )
        frequencies = torch.exp(exponent / max(half_dim - 1, 1))

        # angles: [B, 1] * [1, D/2] -> [B, D/2]。
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)  # [B, 2*(D/2)]

        # embedding_dim 为奇数时在末尾补一个 0，保证输出严格为 [B, D]。
        if embedding.shape[1] < self.embedding_dim:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding
