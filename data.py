"""ImageFolder 数据读取。类别标签可作为 DiT 的生成条件。"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent


def _seed_worker(worker_id: int) -> None:
    """让每个 DataLoader 子进程的 Python/NumPy 随机数也可复现。"""
    del worker_id  # worker 的基础 seed 已由 PyTorch 设置，不需要直接使用编号。
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloader(
    dataset_config: dict,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, Path]:
    """创建训练 DataLoader。

    ImageFolder 期望目录结构为 root/class_name/image.JPEG。imagenet-10 正好
    是这种结构；它只接受已知图像扩展名，因此会自动忽略 Zone.Identifier 文件。

    每个 batch 返回：
        images: [B, 3, H, W]，float32，范围 [-1, 1]
        labels: [B]，ImageFolder 按类别目录名排序得到的类别编号
    """
    configured_path = Path(dataset_config["path"]).expanduser()
    dataset_path = (
        configured_path
        if configured_path.is_absolute()
        else (PROJECT_ROOT / configured_path).resolve()
    )
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"找不到数据集目录：{dataset_path}\n"
            "请修改配置中的 dataset.path，使其指向包含类别子目录的 imagenet-10。"
        )

    image_size = int(dataset_config["image_size"])
    transform_steps: list[object] = [
        # Resize(image_size) 保持长宽比，让较短边变为 image_size。
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
    ]
    if bool(dataset_config["random_horizontal_flip"]):
        transform_steps.append(transforms.RandomHorizontalFlip())
    transform_steps.extend(
        [
            transforms.ToTensor(),  # PIL -> [3,H,W]，范围 [0,1]
            # DDPM 通常在 [-1,1] 上训练：(pixel - 0.5) / 0.5。
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )

    dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=transforms.Compose(transform_steps),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"数据集 {dataset_path} 中没有可读取的图像")

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=int(dataset_config["batch_size"]),
        shuffle=True,
        num_workers=int(dataset_config["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=int(dataset_config["num_workers"]) > 0,
    )
    return dataloader, dataset_path
