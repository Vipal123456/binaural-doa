"""DP-RTF Learning baseline adapted to the current KEMAR benchmark."""

from __future__ import annotations

from typing import Dict

import scipy.io
import torch
import torch.nn as nn


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False,
    )


class _BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, use_res: bool = False):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.use_res = use_res and inplanes == planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_res:
            out = out + residual
        return self.relu(out)


class DPRTFRTFLearn(nn.Module):
    """Port of the paper's CRNN_our front-end with device-safe code."""

    def __init__(
        self,
        planes: int = 64,
        rnn_in_dim: int = 256,
        rnn_hid_dim: int = 256,
        rnn_out_dim: int = 384,
        use_residual_blocks: bool = False,
        dropout: float = 0.4,
    ):
        super().__init__()
        half_planes = planes // 2
        self.layer_gate = nn.Sequential(
            _conv3x3(2, half_planes),
            nn.BatchNorm2d(half_planes),
            nn.ReLU(inplace=True),
            _conv3x3(half_planes, half_planes),
            nn.BatchNorm2d(half_planes),
            nn.Sigmoid(),
        )
        self.layer_dm = _conv1x1(2, half_planes)
        self.layer_dm_2 = nn.Sequential(
            _conv1x1(half_planes, half_planes),
            nn.BatchNorm2d(half_planes),
            nn.ReLU(inplace=True),
        )
        self.layer_dp = _conv1x1(2, half_planes)
        self.layer_dp_2 = nn.Sequential(
            _conv1x1(planes, half_planes),
            nn.BatchNorm2d(half_planes),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            _BasicBlock(planes, planes, use_res=use_residual_blocks),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.layer2 = nn.Sequential(
            _BasicBlock(planes, planes, use_res=use_residual_blocks),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.layer3 = nn.Sequential(
            _BasicBlock(planes, planes, use_res=use_residual_blocks),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.layer4 = nn.Sequential(
            _BasicBlock(planes, planes, use_res=use_residual_blocks),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.layer5 = nn.Sequential(
            _BasicBlock(planes, planes, use_res=use_residual_blocks),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.rnn = nn.GRU(
            input_size=rnn_in_dim,
            hidden_size=rnn_hid_dim,
            num_layers=1,
            batch_first=True,
            bias=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.rnn_fc = nn.Sequential(
            nn.Linear(rnn_hid_dim, rnn_out_dim),
            nn.Tanh(),
        )

    def forward(self, mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        gate = self.layer_gate(mag)

        fea_magd = torch.tanh(self.layer_dm(mag))
        fea_magd = self.layer_dm_2(fea_magd)

        fea_phased = self.layer_dp(phase)
        fea_phased = torch.cat((torch.sin(fea_phased), torch.cos(fea_phased)), dim=1)
        fea_phased = self.layer_dp_2(fea_phased)

        fea = torch.cat((fea_magd * gate, fea_phased * gate), dim=1)
        fea = self.layer1(fea)
        fea = self.layer2(fea)
        fea = self.layer3(fea)
        fea = self.layer4(fea)
        fea = self.layer5(fea)

        fea_cnn = fea.reshape(fea.size(0), -1, fea.size(3)).permute(0, 2, 1)
        fea_rnn, _ = self.rnn(fea_cnn)
        fea_rnn_fc = self.rnn_fc(fea_rnn[:, -1, :])
        return fea_rnn_fc.unsqueeze(-1)


class DPRTFKemarDOANet(nn.Module):
    """DP-RTF baseline using current static-dataset feature tensors."""

    def __init__(
        self,
        template_path: str,
        num_classes: int = 72,
        freq_bins_used: int = 128,
        planes: int = 64,
        rnn_hidden_size: int = 256,
        use_residual_blocks: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.freq_bins_used = int(freq_bins_used)

        rtf_dp = scipy.io.loadmat(template_path)["rtf_dp"].astype("float32")
        template = torch.from_numpy(rtf_dp)
        if template.ndim != 2:
            raise ValueError(f"Expected rtf_dp to be 2D, got shape {tuple(template.shape)}")
        if template.shape[1] != self.num_classes:
            raise ValueError(
                f"Template DOA count mismatch: template has {template.shape[1]}, "
                f"config expects {self.num_classes}"
            )

        self.rtf_dim = int(template.shape[0])
        expected_dim = self.freq_bins_used * 3
        if self.rtf_dim != expected_dim:
            raise ValueError(
                f"Template feature dim mismatch: template has {self.rtf_dim}, "
                f"expected {expected_dim} for freq_bins_used={self.freq_bins_used}"
            )

        self.register_buffer("rtf_template_set", template, persistent=True)
        self.rtflearn_block = DPRTFRTFLearn(
            planes=planes,
            rnn_in_dim=4 * planes,
            rnn_hid_dim=rnn_hidden_size,
            rnn_out_dim=self.rtf_dim,
            use_residual_blocks=use_residual_blocks,
        )

    def _build_mag_phase_tensors(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        real_l = batch["spec_real_L"][..., : self.freq_bins_used]
        imag_l = batch["spec_imag_L"][..., : self.freq_bins_used]
        real_r = batch["spec_real_R"][..., : self.freq_bins_used]
        imag_r = batch["spec_imag_R"][..., : self.freq_bins_used]

        mag_l = torch.sqrt(real_l.square() + imag_l.square() + 1e-8)
        mag_r = torch.sqrt(real_r.square() + imag_r.square() + 1e-8)
        phase_l = torch.atan2(imag_l, real_l)
        phase_r = torch.atan2(imag_r, real_r)

        mag = torch.stack([mag_l, mag_r], dim=1).permute(0, 1, 3, 2).contiguous()
        phase = torch.stack([phase_l, phase_r], dim=1).permute(0, 1, 3, 2).contiguous()
        return mag, phase

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mag, phase = self._build_mag_phase_tensors(batch)
        mag_log = torch.log10(mag + 1e-5)

        rtf_feat = self.rtflearn_block(mag_log, phase)  # [B, 3F, 1]
        template = self.rtf_template_set.unsqueeze(0).expand(rtf_feat.size(0), -1, -1)
        sim = -torch.square(template - rtf_feat.expand(-1, -1, template.size(2))).sum(dim=1)

        return {
            "logits": sim,
            "rtf_feat": rtf_feat,
            "rtf_template_set": template,
        }
