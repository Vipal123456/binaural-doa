"""可视化辅助工具（混淆矩阵等）。"""

import os
from typing import Optional

import numpy as np


def save_confusion_matrix(cm: np.ndarray,
                          save_path: str,
                          title: str = "Confusion Matrix",
                          labels: Optional[list] = None) -> None:
    """将混淆矩阵热图保存为图片。

    使用 matplotlib；如果未安装 matplotlib 则静默跳过。

    参数:
        cm: 形状为 ``(C, C)`` 的混淆矩阵。
        save_path: 输出图片路径（例如 ``"outputs/cm.png"``）。
        title: 图表标题。
        labels: 可选的刻度标签。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    if labels is not None:
        tick_pos = np.arange(len(labels))
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
