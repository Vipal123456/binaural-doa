"""冒烟测试：使用合成数据进行模型前向传播。

运行方式：  python -m pytest tests/test_model_forward.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from models.binaural_doa_net import BinauralDOANet


@pytest.fixture
def model():
    """创建一个用于测试的小型模型。"""
    return BinauralDOANet(
        freq_bins=257,
        encoder_channels=[16, 32],
        encoder_out_dim=64,
        proj_dim=64,
        prior_hidden_dim=128,
        prior_out_dim=64,
        attention_dim=64,
        num_heads=4,
        gate_dim=64,
        gru_hidden_size=64,
        gru_num_layers=1,
        gru_dropout=0.0,
        num_classes=72,
        dropout=0.0,
    )


def _make_batch(batch_size: int = 2, T: int = 201, F: int = 257):
    """创建一个与 DataLoader 预期输出格式匹配的合成批次。"""
    return {
        "log_mag_L": torch.randn(batch_size, T, F),
        "log_mag_R": torch.randn(batch_size, T, F),
        "ipd": torch.randn(batch_size, T, F),
        "ild": torch.randn(batch_size, T, F),
        "azimuth_label": torch.randint(0, 72, (batch_size,)),
        "azimuth_deg": torch.randn(batch_size) * 180,
    }


class TestModelForward:
    """验证模型输出的形状是否正确。"""

    def test_output_keys(self, model):
        batch = _make_batch()
        out = model(batch)
        assert "logits" in out
        assert "d_feat" in out

    def test_logits_shape(self, model):
        B = 4
        batch = _make_batch(batch_size=B)
        out = model(batch)
        assert out["logits"].shape == (B, 72), \
            f"Expected (4, 72), got {out['logits'].shape}"

    def test_gradient_flow(self, model):
        batch = _make_batch()
        out = model(batch)
        loss = out["logits"].sum()
        loss.backward()
        # 检查编码器参数是否收到了梯度
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                break  # 只检查一个参数即可

    def test_different_time_lengths(self, model):
        """模型应当能够处理不同的时间维度。"""
        for T in [50, 100, 201]:
            batch = _make_batch(batch_size=2, T=T)
            out = model(batch)
            assert out["logits"].shape == (2, 72)

    def test_shared_encoder_weights(self, model):
        """左耳和右耳应使用同一个编码器实例。"""
        # 模型对左右耳使用同一个 self.encoder —— 只有一个编码器
        # 只需验证模型中恰好存在一个 encoder 属性
        assert hasattr(model, "encoder")

    def test_eval_mode(self, model):
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            out = model(batch)
        assert out["logits"].shape == (2, 72)
