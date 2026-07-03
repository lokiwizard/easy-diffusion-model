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
    return parser.parse_args()


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
    print(
        f"开始采样：model={config['model']['name']}，"
        f"pred_type={diffusion.pred_type}，shape={shape}，device={device}"
    )
    images = diffusion.sample(model, shape, device)  # [B,3,H,W]
    save_image_grid(images, args.output)
    print(f"图像已保存到：{Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
