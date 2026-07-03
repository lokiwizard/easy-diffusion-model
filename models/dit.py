"""带类别条件和 AdaLN-Zero 的精简 Diffusion Transformer。"""

import torch
from torch import nn

from models.common import SinusoidalTimeEmbedding


def _modulate(
    features: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """用每张图的条件向量调制所有 patch token。"""
    return features * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
    """通过 AdaLN-Zero 把时间与类别条件注入 attention 和 MLP。"""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        # 6 组参数分别控制 attention/MLP 的 shift、scale 和残差 gate。
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        shift_attention, scale_attention, gate_attention, shift_mlp, scale_mlp, gate_mlp = (
            self.ada_ln(condition).chunk(6, dim=1)
        )

        attention_input = _modulate(
            self.norm_attention(tokens),
            shift_attention,
            scale_attention,
        )
        attention_output = self.attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )[0]
        tokens = tokens + gate_attention[:, None, :] * self.attention_dropout(
            attention_output
        )

        mlp_input = _modulate(self.norm_mlp(tokens), shift_mlp, scale_mlp)
        return tokens + gate_mlp[:, None, :] * self.mlp(mlp_input)


class FinalLayer(nn.Module):
    """条件归一化后把每个 token 投影成一个像素 patch。"""

    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.projection = nn.Linear(
            hidden_dim,
            patch_size * patch_size * out_channels,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.ada_ln(condition).chunk(2, dim=1)
        return self.projection(_modulate(self.norm(tokens), shift, scale))


class SimpleDiT(nn.Module):
    """把图像切成 patch，并以时间步和 ImageFolder 类别为条件预测扩散目标。

    `num_classes` 个真实类别之外额外保留一个空类别。训练时随机把真实标签替换为空
    类别，使同一个网络同时学会条件与无条件预测，供 CFG 采样使用。
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
        num_classes: int = 10,
        class_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size 必须能被 patch_size 整除")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        if num_classes < 1:
            raise ValueError("num_classes 必须大于 0")
        if not 0.0 <= class_dropout_prob <= 1.0:
            raise ValueError("class_dropout_prob 必须在 [0,1] 内")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.null_class = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2

        # [B,C,H,W] -> [B,D,H/P,W/P]；每个卷积窗口对应一个不重叠 patch。
        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_dim)
        )
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        # 最后一项是 CFG 使用的空类别，不对应数据集中的真实目录。
        self.class_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_dim, patch_size, in_channels)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.class_embedding.weight, std=0.02)

        # AdaLN-Zero 让每个 block 初始等价于恒等映射，输出层初始预测为 0。
        for block in self.blocks:
            nn.init.zeros_(block.ada_ln[-1].weight)
            nn.init.zeros_(block.ada_ln[-1].bias)
        nn.init.zeros_(self.final_layer.ada_ln[-1].weight)
        nn.init.zeros_(self.final_layer.ada_ln[-1].bias)
        nn.init.zeros_(self.final_layer.projection.weight)
        nn.init.zeros_(self.final_layer.projection.bias)

    def _prepare_class_labels(
        self,
        class_labels: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """校验标签，并在训练时执行 classifier-free label dropout。"""
        if class_labels is None:
            return torch.full(
                (batch_size,),
                self.null_class,
                device=device,
                dtype=torch.long,
            )
        if class_labels.shape != (batch_size,):
            raise ValueError(
                f"class_labels 应为 shape {(batch_size,)}，实际为 {tuple(class_labels.shape)}"
            )
        if class_labels.dtype != torch.long:
            raise TypeError("class_labels 必须是 torch.long")
        if class_labels.device != device:
            raise ValueError("class_labels 与输入图像必须位于同一设备")
        if torch.any((class_labels < 0) | (class_labels >= self.num_classes)):
            raise ValueError(f"class_labels 必须在 [0,{self.num_classes - 1}] 内")

        if self.training and self.class_dropout_prob > 0:
            drop_mask = torch.rand(batch_size, device=device) < self.class_dropout_prob
            null_labels = torch.full_like(class_labels, self.null_class)
            class_labels = torch.where(drop_mask, null_labels, class_labels)
        return class_labels

    def _unpatchify(self, patch_values: torch.Tensor) -> torch.Tensor:
        """[B,N,P*P*C] -> [B,C,H,W]。"""
        batch_size = patch_values.shape[0]
        grid = self.grid_size
        patch = self.patch_size
        channels = self.in_channels

        images = patch_values.reshape(batch_size, grid, grid, patch, patch, channels)
        images = images.permute(0, 5, 1, 3, 2, 4).contiguous()
        return images.reshape(
            batch_size, channels, self.image_size, self.image_size
        )

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """输入 x_t [B,C,H,W]、t [B] 和可选类别 y [B]，输出 [B,C,H,W]。"""
        expected_hw = (self.image_size, self.image_size)
        if tuple(noisy_images.shape[-2:]) != expected_hw:
            raise ValueError(
                f"DiT 固定输入尺寸为 {expected_hw}，实际为 {tuple(noisy_images.shape[-2:])}"
            )
        batch_size = noisy_images.shape[0]
        if timesteps.shape != (batch_size,):
            raise ValueError(
                f"timesteps 应为 shape {(batch_size,)}，实际为 {tuple(timesteps.shape)}"
            )

        patch_features = self.patch_embed(noisy_images)
        tokens = patch_features.flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding

        labels = self._prepare_class_labels(
            class_labels,
            batch_size,
            noisy_images.device,
        )
        condition = self.time_mlp(timesteps) + self.class_embedding(labels)
        for block in self.blocks:
            tokens = block(tokens, condition)

        patch_values = self.final_layer(tokens, condition)
        return self._unpatchify(patch_values)
