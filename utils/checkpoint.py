"""检查点保存 / 加载工具。"""

import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(state: Dict[str, Any],
                    save_dir: str,
                    filename: str = "latest.pth") -> str:
    """保存训练检查点。

    参数:
        state: 包含 model state_dict、optimizer 状态、epoch 等的字典。
        save_dir: 检查点文件写入目录。
        filename: 检查点文件名。

    返回:
        保存的检查点完整路径。
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(path: str,
                    map_location: Optional[str] = None) -> Dict[str, Any]:
    """从磁盘加载检查点。

    参数:
        path: ``.pth`` 文件路径。
        map_location: 设备映射（例如 ``"cpu"``）。

    返回:
        加载的状态字典。

    异常:
        FileNotFoundError: 当 *path* 不存在时。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"检查点未找到: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)
