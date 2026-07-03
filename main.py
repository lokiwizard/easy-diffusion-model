"""兼容入口：执行此文件等价于执行 train.py。"""

from train import main


if __name__ == "__main__":  # pragma: no cover - 这里只做入口转发
    main()
