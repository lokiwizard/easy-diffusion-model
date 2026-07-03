"""DDPM 训练入口。

训练循环有意保持直白：取数据 -> 随机时间步 -> 加噪并计算目标 -> 反传 -> 保存。
没有 Trainer 类或回调系统，便于新手逐行跟踪一次参数更新发生了什么。
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import torch
import yaml
from tqdm.auto import tqdm

from data import build_dataloader
from diffusion import GaussianDiffusion
from models import build_model
from utils import (
    config_hash,
    load_config,
    resolve_device,
    save_config,
    save_image_grid,
    set_random_seed,
    validate_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练一个简单的 DDPM 图像生成模型")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖一个配置项；可重复使用，例如 --set diffusion.pred_type=v",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="从 last.pt 或 epoch_xxxx.pt 断点继续",
    )
    return parser.parse_args()


def print_training_config(config: dict, device: torch.device, dataset_path: Path) -> None:
    """在训练开始时打印用户要求的关键配置和完整 YAML。"""
    dataset = config["dataset"]
    diffusion = config["diffusion"]
    training = config["training"]
    print("\n========== 当前训练配置 ==========")
    print(f"dataset       : {dataset_path}")
    print(f"model         : {config['model']['name']}")
    print(f"pred_type     : {diffusion['pred_type']}")
    print(f"image_size    : {dataset['image_size']}")
    print(f"batch_size    : {dataset['batch_size']}")
    print(f"timesteps     : {diffusion['timesteps']}")
    print(f"learning_rate : {training['learning_rate']}")
    print(f"epochs        : {training['epochs']}")
    print(f"device        : {device}")
    print("------------- 完整 YAML ------------")
    print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True).rstrip())
    print("====================================\n")


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict,
    class_names: list[str],
    epoch: int,
    global_step: int,
) -> None:
    """保存恢复训练所需的全部状态。"""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "class_names": class_names,
            "epoch": epoch,
            "global_step": global_step,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    current_config: dict,
    current_class_names: list[str],
) -> tuple[int, int]:
    """恢复模型与优化器，返回 (下一个 epoch, 已完成 step 数)。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_config = checkpoint["config"]

    model_name = current_config["model"]["name"]
    # 只比较当前启用的模型配置；修改未启用模型不应阻止安全恢复。
    critical_pairs = [
        ("model.name", saved_config["model"]["name"], model_name),
        (
            "model.in_channels",
            saved_config["model"]["in_channels"],
            current_config["model"]["in_channels"],
        ),
        (
            f"model.{model_name}",
            saved_config["model"][model_name],
            current_config["model"][model_name],
        ),
        (
            "dataset.image_size",
            saved_config["dataset"]["image_size"],
            current_config["dataset"]["image_size"],
        ),
        ("diffusion", saved_config["diffusion"], current_config["diffusion"]),
    ]
    for name, saved_value, current_value in critical_pairs:
        if saved_value != current_value:
            raise ValueError(f"断点中的 {name} 与当前配置不同，不能安全恢复训练")
    saved_class_names = checkpoint.get("class_names")
    if saved_class_names is not None and saved_class_names != current_class_names:
        raise ValueError("断点中的类别名称或顺序与当前数据集不同，不能安全恢复训练")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    validate_config(config)

    seed = int(config["seed"])
    set_random_seed(seed)
    device = resolve_device(config["device"])

    dataloader, dataset_path = build_dataloader(config["dataset"], seed, device)
    if len(dataloader) == 0:
        raise RuntimeError("batch_size 大于数据集大小，导致没有可训练的 batch")
    class_names = list(dataloader.dataset.classes)
    is_class_conditional = config["model"]["name"] == "dit"
    if is_class_conditional:
        configured_num_classes = int(config["model"]["dit"]["num_classes"])
        if len(class_names) != configured_num_classes:
            raise ValueError(
                "model.dit.num_classes 与数据集类别数不一致："
                f"配置为 {configured_num_classes}，数据集为 {len(class_names)}"
            )
    print_training_config(config, device, dataset_path)
    print(f"读取到 {len(dataloader.dataset):,} 张图像、{len(class_names)} 个目录类别")
    if is_class_conditional:
        print("类别映射：" + ", ".join(f"{index}={name}" for index, name in enumerate(class_names)))

    image_size = int(config["dataset"]["image_size"])
    model = build_model(config["model"], image_size).to(device)
    diffusion_config = config["diffusion"]
    diffusion = GaussianDiffusion(
        timesteps=int(diffusion_config["timesteps"]),
        beta_start=float(diffusion_config["beta_start"]),
        beta_end=float(diffusion_config["beta_end"]),
        pred_type=diffusion_config["pred_type"],
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"模型参数量：{parameter_count / 1e6:.2f} M")

    training_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    use_amp = bool(training_config["use_amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"混合精度训练：{'开启' if use_amp else '关闭'}")

    if args.resume:
        checkpoint_path = Path(args.resume).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到断点：{checkpoint_path}")
        run_directory = checkpoint_path.parent
        start_epoch, global_step = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scaler,
            config,
            class_names,
        )
        print(f"已从 {checkpoint_path} 恢复，将从 epoch {start_epoch + 1} 继续")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = (
            f"{config['model']['name']}_{diffusion_config['pred_type']}_"
            f"{timestamp}_{config_hash(config)}"
        )
        run_directory = Path(config["output_dir"]) / experiment_name
        run_directory.mkdir(parents=True, exist_ok=False)
        save_config(config, run_directory / "config.yaml")
        start_epoch = 0
        global_step = 0

    metrics_path = run_directory / "metrics.csv"
    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
            csv.writer(metrics_file).writerow(["epoch", "global_step", "loss", "learning_rate"])

    epochs = int(training_config["epochs"])
    max_steps_value = training_config["max_steps"]
    max_steps = None if max_steps_value is None else int(max_steps_value)
    should_stop = max_steps is not None and global_step >= max_steps
    training_start_time = time.time()

    for epoch in range(start_epoch, epochs):
        if should_stop:
            break
        model.train()
        epoch_loss_sum = 0.0
        steps_in_epoch = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for clean_images, labels in progress:
            # clean_images/x_0: [B,3,H,W]，值域 [-1,1]。
            clean_images = clean_images.to(device, non_blocking=True)
            class_labels = (
                labels.to(device, non_blocking=True)
                if is_class_conditional
                else None
            )
            batch_size = clean_images.shape[0]
            # 每张图独立均匀抽一个时间步；timesteps shape 为 [B]。
            timesteps = torch.randint(
                low=0,
                high=diffusion.timesteps,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )

            optimizer.zero_grad(set_to_none=True)
            # AMP 只在 CUDA 上启用；CPU 路径保持普通 float32，兼容性更好。
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                loss = diffusion.training_loss(
                    model,
                    clean_images,
                    timesteps,
                    class_labels=class_labels,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(training_config["grad_clip"])
            )
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            steps_in_epoch += 1
            loss_value = float(loss.detach())
            epoch_loss_sum += loss_value
            progress.set_postfix(loss=f"{loss_value:.4f}", step=global_step)

            log_every = int(training_config["log_every_steps"])
            if global_step % log_every == 0:
                with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
                    csv.writer(metrics_file).writerow(
                        [
                            epoch + 1,
                            global_step,
                            f"{loss_value:.8f}",
                            optimizer.param_groups[0]["lr"],
                        ]
                    )

            if max_steps is not None and global_step >= max_steps:
                should_stop = True
                break

        mean_epoch_loss = epoch_loss_sum / max(steps_in_epoch, 1)
        print(f"Epoch {epoch + 1} 平均 loss：{mean_epoch_loss:.6f}")

        # last.pt 每个 epoch 更新，意外中断时至多损失当前 epoch 的进度。
        save_checkpoint(
            run_directory / "last.pt",
            model,
            optimizer,
            scaler,
            config,
            class_names,
            epoch,
            global_step,
        )
        save_every = int(training_config["save_every_epochs"])
        if save_every > 0 and (epoch + 1) % save_every == 0:
            save_checkpoint(
                run_directory / f"epoch_{epoch + 1:04d}.pt",
                model,
                optimizer,
                scaler,
                config,
                class_names,
                epoch,
                global_step,
            )

        sample_every = int(training_config["sample_every_epochs"])
        if sample_every > 0 and (epoch + 1) % sample_every == 0:
            sample_count = int(training_config["sample_count"])
            sample_shape = (
                sample_count,
                int(config["model"]["in_channels"]),
                image_size,
                image_size,
            )
            sample_labels = None
            cfg_scale = 1.0
            if is_class_conditional:
                sample_labels = torch.arange(sample_count, device=device)
                sample_labels = sample_labels % len(class_names)
                cfg_scale = float(config["sampling"]["cfg_scale"])
            samples = diffusion.sample(
                model,
                sample_shape,
                device,
                class_labels=sample_labels,
                cfg_scale=cfg_scale,
            )
            save_image_grid(samples, run_directory / f"samples_epoch_{epoch + 1:04d}.png")

    elapsed_minutes = (time.time() - training_start_time) / 60.0
    print(f"\n训练结束。global_step={global_step}，耗时={elapsed_minutes:.2f} 分钟")
    print(f"实验产物：{run_directory.resolve()}")


if __name__ == "__main__":
    main()
