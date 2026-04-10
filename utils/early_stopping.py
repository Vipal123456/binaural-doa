"""早停机制：监控验证指标，在没有改善时提前停止训练。"""

import numpy as np


class EarlyStopping:
    """早停机制。

    当验证指标在指定轮数内没有改善时，停止训练以防止过拟合。

    参数
    ----------
    patience : int
        验证指标无改善时等待的轮数。
    delta : float
        最小改善阈值（低于此值视为无改善）。
    mode : str
        'min' 表示指标越小越好（如loss、MAE），
        'max' 表示指标越大越好（如accuracy）。
    verbose : bool
        是否打印详细信息。
    """

    def __init__(self, patience=7, delta=0.0, mode='min', verbose=True):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_value = np.inf if mode == 'min' else -np.inf

    def __call__(self, current_value, epoch=None):
        """检查当前指标值，更新早停状态。

        参数
        ----------
        current_value : float
            当前轮的验证指标值。
        epoch : int, optional
            当前轮数（用于日志）。

        返回
        -------
        bool
            如果应该早停，返回True；否则返回False。
        """
        if self.mode == 'min':
            score = -current_value
        else:
            score = current_value

        if self.best_score is None:
            # 第一次调用
            self.best_score = score
            self.best_value = current_value
            if self.verbose:
                msg = f"[EarlyStopping] 初始最佳值: {current_value:.4f}"
                if epoch is not None:
                    msg += f" (Epoch {epoch})"
                print(msg)
        elif score < self.best_score + self.delta:
            # 没有改善
            self.counter += 1
            if self.verbose:
                msg = f"[EarlyStopping] 无改善计数: {self.counter}/{self.patience} (当前: {current_value:.4f}, 最佳: {self.best_value:.4f})"
                if epoch is not None:
                    msg += f" (Epoch {epoch})"
                print(msg)
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"[EarlyStopping] 触发早停！最佳值: {self.best_value:.4f}")
        else:
            # 有改善
            if self.verbose and self.best_value != current_value:
                improvement = abs(current_value - self.best_value)
                msg = f"[EarlyStopping] 指标改善 {improvement:.4f} ({self.best_value:.4f} → {current_value:.4f})"
                if epoch is not None:
                    msg += f" (Epoch {epoch})"
                print(msg)
            self.best_score = score
            self.best_value = current_value
            self.counter = 0

        return self.early_stop

    def state_dict(self):
        """返回早停状态字典（用于保存checkpoint）。"""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_value': self.best_value,
            'early_stop': self.early_stop,
        }

    def load_state_dict(self, state_dict):
        """从状态字典恢复早停状态（用于恢复训练）。"""
        self.counter = state_dict.get('counter', 0)
        self.best_score = state_dict.get('best_score', None)
        self.best_value = state_dict.get('best_value',
                                          np.inf if self.mode == 'min' else -np.inf)
        self.early_stop = state_dict.get('early_stop', False)
