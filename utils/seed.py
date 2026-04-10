"""固定随机种子以保证可复现性。"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """统一固定 Python、NumPy 和 PyTorch 的随机种子。

    参数:
        seed: 整数种子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确定性行为（可能降低速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
