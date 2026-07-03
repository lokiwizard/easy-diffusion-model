"""模型构建入口。保持显式分支，方便新手定位具体实现。"""

from torch import nn

from models.dit import SimpleDiT
from models.unet import SimpleUNet


def build_model(model_config: dict, image_size: int) -> nn.Module:
    """根据 YAML 中的 model.name 创建模型。"""
    model_name = model_config["name"].lower()
    in_channels = int(model_config["in_channels"])

    if model_name == "unet":
        unet_config = model_config["unet"]
        return SimpleUNet(
            in_channels=in_channels,
            base_channels=int(unet_config["base_channels"]),
            time_embed_dim=int(unet_config["time_embed_dim"]),
        )

    if model_name == "dit":
        dit_config = model_config["dit"]
        return SimpleDiT(
            image_size=image_size,
            patch_size=int(dit_config["patch_size"]),
            in_channels=in_channels,
            hidden_dim=int(dit_config["hidden_dim"]),
            depth=int(dit_config["depth"]),
            num_heads=int(dit_config["num_heads"]),
            mlp_ratio=float(dit_config["mlp_ratio"]),
            dropout=float(dit_config["dropout"]),
        )

    raise ValueError(f"未知模型 {model_name!r}，可选值为 'unet' 或 'dit'")
