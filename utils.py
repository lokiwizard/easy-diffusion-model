"""配置、随机种子、设备和图像保存等小工具。"""

import hashlib
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.utils import save_image


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """应用形如 `diffusion.pred_type=v` 的命令行覆盖，并拒绝拼错的键。"""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set 参数必须是 key=value，实际为 {item!r}")
        dotted_key, raw_value = item.split("=", maxsplit=1)
        keys = dotted_key.split(".")
        target = config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise KeyError(f"配置中不存在 {dotted_key!r}")
            target = target[key]
        final_key = keys[-1]
        if final_key not in target:
            raise KeyError(f"配置中不存在 {dotted_key!r}")
        target[final_key] = yaml.safe_load(raw_value)
    return config


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> dict:
    """读取 YAML，并应用可选命令行覆盖。"""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("配置文件顶层必须是字典")
    return apply_overrides(config, overrides or [])


def validate_config(config: dict) -> None:
    """尽早检查关键配置，避免训练数分钟后才暴露简单拼写错误。"""
    required_sections = {"dataset", "model", "diffusion", "training"}
    missing = required_sections - config.keys()
    if missing:
        raise KeyError(f"配置缺少顶层字段：{sorted(missing)}")

    if config["model"]["name"] not in {"unet", "dit"}:
        raise ValueError("model.name 必须是 unet 或 dit")
    if config["diffusion"]["pred_type"] not in {"epsilon", "x0", "v", "score"}:
        raise ValueError("diffusion.pred_type 必须是 epsilon、x0、v 或 score")
    if int(config["dataset"]["batch_size"]) < 1:
        raise ValueError("dataset.batch_size 必须大于 0")
    if int(config["dataset"]["image_size"]) < 8:
        raise ValueError("dataset.image_size 至少为 8")
    if float(config["training"]["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate 必须大于 0")
    if int(config["training"]["epochs"]) < 1:
        raise ValueError("training.epochs 必须大于 0")


def resolve_device(device_name: str) -> torch.device:
    """把 auto/cuda/cpu 配置转换成 torch.device。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但 torch.cuda.is_available() 为 False")
    if device_name not in {"cuda", "cpu"}:
        raise ValueError("device 必须是 auto、cuda 或 cpu")
    return torch.device(device_name)


def set_random_seed(seed: int) -> None:
    """固定 Python、NumPy、CPU 和 CUDA 随机数。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 确定性卷积便于复现实验；代价是某些 GPU 上训练会略慢。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config_hash(config: dict) -> str:
    """返回配置内容的 8 位哈希，用于区分实验。"""
    serialized = yaml.safe_dump(config, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]


def save_config(config: dict, output_path: Path) -> None:
    """把本次实际使用的完整配置写入运行目录。"""
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)


def save_image_grid(images: torch.Tensor, output_path: str | Path) -> None:
    """把 [B,3,H,W]、[-1,1] 图像保存成方形网格。"""
    images_01 = (images.detach().cpu().clamp(-1, 1) + 1.0) / 2.0
    images_per_row = max(1, int(math.sqrt(images_01.shape[0])))
    save_image(images_01, output_path, nrow=images_per_row)
