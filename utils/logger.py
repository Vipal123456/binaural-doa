"""日志工具：终端输出 + 文件记录 + 可选 TensorBoard。"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "DOA-net",
                 log_dir: str = "outputs/logs",
                 level: int = logging.INFO) -> logging.Logger:
    """创建一个同时写入终端和时间戳日志文件的 Logger。

    参数:
        name: Logger 名称。
        log_dir: 日志文件存放目录。
        level: 日志级别。

    返回:
        配置好的 ``logging.Logger`` 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 多次调用时防止重复添加 handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 终端 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件 handler
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"{timestamp}.log"))
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 固定文件名，便于 `tail -f outputs/.../train.log` 实时查看。
    fh_latest = logging.FileHandler(os.path.join(log_dir, "train.log"), mode="w")
    fh_latest.setLevel(level)
    fh_latest.setFormatter(formatter)
    logger.addHandler(fh_latest)

    return logger


class TBWriter:
    """``torch.utils.tensorboard.SummaryWriter`` 的轻量包装器。

    延迟创建 writer，即使未安装 TensorBoard 也不会导致导入失败。
    """

    def __init__(self, log_dir: str = "outputs/logs/tb"):
        self._log_dir = log_dir
        self._writer: Optional[object] = None

    def _ensure_writer(self):
        if self._writer is None:
            os.makedirs(self._log_dir, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(self._log_dir)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._ensure_writer()
        self._writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int) -> None:
        self._ensure_writer()
        self._writer.add_scalars(main_tag, tag_scalar_dict, step)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
