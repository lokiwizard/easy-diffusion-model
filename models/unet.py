"""一个适合入门阅读的、带时间条件的卷积 UNet。"""

import torch
import torch.nn.functional as F
from torch import nn

from models.common import SinusoidalTimeEmbedding


def _valid_group_count(channels: int, preferred_groups: int = 8) -> int:
    """选择能整除通道数的 GroupNorm 组数。"""
    groups = min(preferred_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class ResidualBlock(nn.Module):
    """带扩散时间条件的残差块。

    图像特征 h 的 shape 是 [B, C_in, H, W]，时间向量的 shape 是
    [B, D_time]。时间向量经线性层变为 [B, C_out]，再扩展为
    [B, C_out, 1, 1] 加到每个空间位置上。
    """

    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_valid_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels),
        )
        self.norm2 = nn.GroupNorm(_valid_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 当输入输出通道数不同，用 1x1 卷积匹配残差分支的 shape。
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, image_features: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """输入 [B,C_in,H,W]，输出 [B,C_out,H,W]。"""
        hidden = self.conv1(F.silu(self.norm1(image_features)))  # [B, C_out, H, W]
        time_bias = self.time_projection(time_embedding)[:, :, None, None]  # [B,C_out,1,1]
        hidden = hidden + time_bias
        hidden = self.conv2(F.silu(self.norm2(hidden)))  # [B, C_out, H, W]
        return hidden + self.skip(image_features)  # [B, C_out, H, W]


class Downsample(nn.Module):
    """高和宽减半、通道数翻倍。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,C,H,W] -> [B,C_out,H/2,W/2]。"""
        return self.conv(x)


class Upsample(nn.Module):
    """高和宽翻倍、通道数降低。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,C,H,W] -> [B,C_out,2H,2W]。"""
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class SimpleUNet(nn.Module):
    """两次下采样的简洁 UNet。

    以默认 64x64 输入为例，空间尺寸依次为：
    64x64 -> 32x32 -> 16x16 -> 32x32 -> 64x64。

    UNet 的关键是把下采样阶段的高分辨率特征通过 skip connection
    拼接给上采样阶段。这样既保留局部细节，又能在低分辨率层建模全局结构。
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        time_embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if base_channels < 8:
            raise ValueError("base_channels 至少应为 8")

        self.in_channels = in_channels
        c = base_channels

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(c),
            nn.Linear(c, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
        self.down_block_1 = ResidualBlock(c, c, time_embed_dim)
        self.downsample_1 = Downsample(c, 2 * c)
        self.down_block_2 = ResidualBlock(2 * c, 2 * c, time_embed_dim)
        self.downsample_2 = Downsample(2 * c, 4 * c)

        self.middle_block_1 = ResidualBlock(4 * c, 4 * c, time_embed_dim)
        self.middle_block_2 = ResidualBlock(4 * c, 4 * c, time_embed_dim)

        self.upsample_2 = Upsample(4 * c, 2 * c)
        # 上采样特征 [B,2C,H/2,W/2] 与 skip_2（同 shape）拼接后是 4C 通道。
        self.up_block_2 = ResidualBlock(4 * c, 2 * c, time_embed_dim)
        self.upsample_1 = Upsample(2 * c, c)
        # 上采样特征 [B,C,H,W] 与 skip_1（同 shape）拼接后是 2C 通道。
        self.up_block_1 = ResidualBlock(2 * c, c, time_embed_dim)

        self.output_norm = nn.GroupNorm(_valid_group_count(c), c)
        self.output_conv = nn.Conv2d(c, in_channels, kernel_size=3, padding=1)

    def forward(self, noisy_images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """预测与输入图像同 shape 的扩散目标。

        Args:
            noisy_images: x_t，shape [B, 3, H, W]，值域通常约为 [-1, 1]。
            timesteps: 每张图对应的离散时间步，shape [B]，整数范围 [0, T-1]。
        Returns:
            model_output: shape [B, 3, H, W]；含义由 pred_type 决定。
        """
        if noisy_images.ndim != 4:
            raise ValueError(f"UNet 输入应为 [B,C,H,W]，实际为 {tuple(noisy_images.shape)}")
        if noisy_images.shape[-2] % 4 != 0 or noisy_images.shape[-1] % 4 != 0:
            raise ValueError("UNet 输入高、宽必须能被 4 整除")

        time_embedding = self.time_mlp(timesteps)  # [B, D_time]

        hidden = self.input_conv(noisy_images)  # [B, C, H, W]
        skip_1 = self.down_block_1(hidden, time_embedding)  # [B, C, H, W]

        hidden = self.downsample_1(skip_1)  # [B, 2C, H/2, W/2]
        skip_2 = self.down_block_2(hidden, time_embedding)  # [B, 2C, H/2, W/2]

        hidden = self.downsample_2(skip_2)  # [B, 4C, H/4, W/4]
        hidden = self.middle_block_1(hidden, time_embedding)  # [B,4C,H/4,W/4]
        hidden = self.middle_block_2(hidden, time_embedding)  # [B,4C,H/4,W/4]

        hidden = self.upsample_2(hidden)  # [B, 2C, H/2, W/2]
        hidden = torch.cat([hidden, skip_2], dim=1)  # [B, 4C, H/2, W/2]
        hidden = self.up_block_2(hidden, time_embedding)  # [B, 2C, H/2, W/2]

        hidden = self.upsample_1(hidden)  # [B, C, H, W]
        hidden = torch.cat([hidden, skip_1], dim=1)  # [B, 2C, H, W]
        hidden = self.up_block_1(hidden, time_embedding)  # [B, C, H, W]

        return self.output_conv(F.silu(self.output_norm(hidden)))  # [B,3,H,W]
