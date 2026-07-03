"""从训练 checkpoint 运行完整 DDPM 采样。"""

import argparse
from pathlib import Path

import torch

from diffusion import GaussianDiffusion
from models import build_model
from utils import resolve_device, save_image_grid, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用训练好的 DDPM 生成图像")
    parser.add_argument("--checkpoint", required=True, help="训练产生的 last.pt 或 epoch_xxxx.pt")
    parser.add_argument("--num-images", type=int, default=16, help="生成图像数量")
    parser.add_argument("--output", default="samples.png", help="输出网格图片路径")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-labels",
        default=None,
        help=(
            "DiT 类别 ID 或类别目录名，逗号分隔；单个值会用于全部图像，"
            "省略时按类别顺序循环"
        ),
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="CFG 强度；省略时读取 checkpoint 配置，1.0 表示不引导",
    )
    return parser.parse_args()


def parse_class_labels(
    raw_labels: str | None,
    num_images: int,
    class_names: list[str],
) -> list[int]:
    """把类别 ID/目录名转换为每张采样图对应的类别 ID。"""
    if raw_labels is None:
        return [index % len(class_names) for index in range(num_images)]

    name_to_index = {name: index for index, name in enumerate(class_names)}
    labels: list[int] = []
    for value in raw_labels.split(","):
        value = value.strip()
        if value in name_to_index:
            labels.append(name_to_index[value])
            continue
        try:
            label = int(value)
        except ValueError as error:
            raise ValueError(f"未知类别 {value!r}") from error
        if not 0 <= label < len(class_names):
            raise ValueError(f"类别 ID 必须在 [0,{len(class_names) - 1}] 内")
        labels.append(label)

    if len(labels) == 1:
        return labels * num_images
    if len(labels) != num_images:
        raise ValueError(
            "--class-labels 应提供一个类别，或为每张图各提供一个类别；"
            f"当前提供 {len(labels)} 个，--num-images={num_images}"
        )
    return labels


def main() -> None:
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("--num-images 必须大于 0")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

    set_random_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    image_size = int(config["dataset"]["image_size"])
    model_name = config["model"]["name"]
    is_class_conditional = model_name == "dit"
    if is_class_conditional and "num_classes" not in config["model"]["dit"]:
        raise ValueError("该 DiT checkpoint 不包含类别条件参数，无法执行类别条件采样")
    model = build_model(config["model"], image_size).to(device)
    model.load_state_dict(checkpoint["model"])

    diffusion_config = config["diffusion"]
    diffusion = GaussianDiffusion(
        timesteps=int(diffusion_config["timesteps"]),
        beta_start=float(diffusion_config["beta_start"]),
        beta_end=float(diffusion_config["beta_end"]),
        pred_type=diffusion_config["pred_type"],
    ).to(device)

    shape = (
        args.num_images,
        int(config["model"]["in_channels"]),
        image_size,
        image_size,
    )
    class_labels = None
    cfg_scale = 1.0
    if is_class_conditional:
        num_classes = int(config["model"]["dit"]["num_classes"])
        class_names = checkpoint.get(
            "class_names",
            [str(index) for index in range(num_classes)],
        )
        if len(class_names) != num_classes:
            raise ValueError("checkpoint 的类别名称数量与模型 num_classes 不一致")
        label_ids = parse_class_labels(args.class_labels, args.num_images, class_names)
        class_labels = torch.tensor(label_ids, device=device, dtype=torch.long)
        configured_cfg_scale = config.get("sampling", {}).get("cfg_scale", 4.0)
        cfg_scale = (
            float(configured_cfg_scale)
            if args.cfg_scale is None
            else args.cfg_scale
        )
        if cfg_scale < 0:
            raise ValueError("--cfg-scale 必须大于等于 0")
        selected_classes = ", ".join(
            f"{label}={class_names[label]}" for label in sorted(set(label_ids))
        )
        print(f"采样类别：{selected_classes}；CFG scale={cfg_scale}")
    elif args.class_labels is not None or (
        args.cfg_scale is not None and args.cfg_scale != 1.0
    ):
        raise ValueError("UNet checkpoint 是无条件模型，不接受类别条件或 CFG")

    print(
        f"开始采样：model={model_name}，"
        f"pred_type={diffusion.pred_type}，shape={shape}，device={device}"
    )
    images = diffusion.sample(
        model,
        shape,
        device,
        class_labels=class_labels,
        cfg_scale=cfg_scale,
    )
    save_image_grid(images, args.output)
    print(f"图像已保存到：{Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
