"""一个精简的 Diffusion Transformer（DiT）实现。"""

import torch
from torch import nn

from models.common import SinusoidalTimeEmbedding


class SimpleDiT(nn.Module):
    """把图像切成 patch，再用 Transformer 预测每个 patch 的扩散目标。

    这是为教学简化后的 DiT：
    1. Conv2d 完成 patchify；
    2. 每个 patch token 加上位置编码和时间编码；
    3. Transformer 在所有 patch 之间交换信息；
    4. 线性层预测每个 patch 的像素，再 unpatchify 回图像。

    它没有类别条件和 AdaLN-Zero，因此比论文版 DiT 更容易先理解和跑通。
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size 必须能被 patch_size 整除")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2

        # [B,3,H,W] -> [B,D,H/P,W/P]；每个卷积窗口正好对应一个不重叠 patch。
        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_dim)
        )  # [1, N, D]
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

        # 每个 token 预测 P*P*C 个数，即一个完整图像 patch。
        self.patch_output = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def _unpatchify(self, patch_values: torch.Tensor) -> torch.Tensor:
        """[B,N,P*P*C] -> [B,C,H,W]。"""
        batch_size = patch_values.shape[0]
        grid = self.grid_size
        patch = self.patch_size
        channels = self.in_channels

        # [B,N,P*P*C] -> [B,grid_h,grid_w,patch_h,patch_w,C]
        images = patch_values.reshape(batch_size, grid, grid, patch, patch, channels)
        # 把通道提前，并让 grid_h/patch_h、grid_w/patch_w 相邻，便于合并维度。
        images = images.permute(0, 5, 1, 3, 2, 4).contiguous()
        return images.reshape(
            batch_size, channels, self.image_size, self.image_size
        )  # [B,C,H,W]

    def forward(self, noisy_images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """输入 x_t [B,3,H,W] 和 t [B]，输出 [B,3,H,W]。"""
        expected_hw = (self.image_size, self.image_size)
        if tuple(noisy_images.shape[-2:]) != expected_hw:
            raise ValueError(
                f"DiT 固定输入尺寸为 {expected_hw}，实际为 {tuple(noisy_images.shape[-2:])}"
            )

        # patch_features: [B,D,grid,grid]。
        patch_features = self.patch_embed(noisy_images)
        # tokens: [B,D,grid*grid] -> [B,N,D]，N 是 patch 数。
        tokens = patch_features.flatten(2).transpose(1, 2)

        time_embedding = self.time_mlp(timesteps)  # [B,D]
        # 同一张图的所有 patch 共享其扩散时间条件；广播后为 [B,N,D]。
        tokens = tokens + self.position_embedding + time_embedding[:, None, :]
        tokens = self.transformer(tokens)  # [B,N,D]
        tokens = self.final_norm(tokens)  # [B,N,D]
        patch_values = self.patch_output(tokens)  # [B,N,P*P*C]
        return self._unpatchify(patch_values)  # [B,C,H,W]
