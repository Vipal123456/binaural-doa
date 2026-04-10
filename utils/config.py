"""配置加载器，支持 YAML 文件和命令行参数覆盖。"""

import argparse
import copy
import os
from typing import Any, Dict, Optional

import yaml


class Config:
    """基于嵌套字典的分层配置管理器。

    支持属性风格访问（``cfg.train.lr``）和
    命令行参数覆盖（``--train.lr 0.0005``）。
    """

    def __init__(self, d: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = d if d is not None else {}

    # ------------------------------------------------------------------
    # 属性访问
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            val = self._data[name]
        except KeyError:
            raise AttributeError(f"Config 中不存在属性 '{name}'")
        if isinstance(val, dict):
            return Config(val)
        return val

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self._data})"

    def to_dict(self) -> Dict[str, Any]:
        """返回一份深拷贝的纯字典。"""
        return copy.deepcopy(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # ------------------------------------------------------------------
    # 扁平键访问  ("train.lr" -> self._data["train"]["lr"])
    # ------------------------------------------------------------------
    def _set_flat(self, flat_key: str, value: Any) -> None:
        """通过点分隔键设置值。"""
        keys = flat_key.split(".")
        d = self._data
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        # 尝试将值转换为原始类型
        old = d.get(keys[-1])
        if old is not None:
            value = _cast(value, type(old))
        d[keys[-1]] = value

    # ------------------------------------------------------------------
    # 输入输出
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 文件加载配置。"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data)

    def save_yaml(self, path: str) -> None:
        """将配置导出为 YAML 文件。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # 命令行覆盖
    # ------------------------------------------------------------------
    def merge_cli(self, argv: Optional[list] = None) -> "Config":
        """从 *argv* 解析 ``--key value`` 键值对并覆盖当前配置。

        返回 *self* 以支持链式调用。
        """
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=str, default=None,
                            help="YAML 配置文件路径")
        known, remaining = parser.parse_known_args(argv)

        # 如果指定了配置文件，用该文件覆盖当前配置
        if known.config is not None:
            override = Config.from_yaml(known.config)
            _deep_update(self._data, override._data)

        # 解析剩余的 --key value 参数对
        i = 0
        while i < len(remaining):
            if remaining[i].startswith("--"):
                key = remaining[i][2:]
                if i + 1 < len(remaining) and not remaining[i + 1].startswith("--"):
                    self._set_flat(key, remaining[i + 1])
                    i += 2
                else:
                    # 标志风格：视为 True
                    self._set_flat(key, True)
                    i += 1
            else:
                i += 1
        return self


# ======================================================================
# 辅助函数
# ======================================================================

def _cast(value: str, target_type: type) -> Any:
    """尽力将 *value*（来自命令行的字符串）转换为 *target_type*。"""
    if target_type == bool:
        return str(value).lower() in ("true", "1", "yes")
    if target_type == list:
        # 接受逗号分隔的值，或者如果已经是列表则保持不变
        if isinstance(value, list):
            return value
        return [v.strip() for v in str(value).split(",")]
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return value


def _deep_update(base: dict, override: dict) -> dict:
    """递归地用 *override* 更新 *base*。"""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(default_path: str = "configs/default.yaml",
                argv: Optional[list] = None) -> Config:
    """便捷函数：加载默认配置，合并命令行覆盖，返回 Config 对象。"""
    cfg = Config.from_yaml(default_path)
    cfg.merge_cli(argv)
    return cfg
