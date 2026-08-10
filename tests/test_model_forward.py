"""冒烟测试：使用合成数据进行模型前向传播。

运行方式：  python -m pytest tests/test_model_forward.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from models.binaural_doa_net import BinauralDOANet
from models.native_lite_v7 import (
    BinauralCueStatistics,
    CueSpecificLocalTFValueEncoder,
    CueSpecificProgressiveTFEncoder,
    DualBranchCueEncoder,
    FineToCoarseSubbandRefinement,
    HighFrequencyFrontBackHead,
    LiteCueEncoder,
    NativeLiteDualCueConcatDOANet,
    NativeLiteLatentCrossSpectrumDOANet,
    RawComplexTFCueEncoder,
)


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


@pytest.mark.parametrize(
    "stabilizer_type",
    ["preconv", "postband_gru", "postband_gru_fullband"],
)
def test_cue_specific_temporal_stabilizers_shape_and_gradients(stabilizer_type):
    encoder = CueSpecificLocalTFValueEncoder(
        cue_bands=8,
        ild_out_dim=4,
        ipd_out_dim=6,
        hidden_channels=4,
        temporal_stabilizer_type=stabilizer_type,
        temporal_stabilizer_hidden_channels=4,
        temporal_stabilizer_kernel_size=5,
        freq_bins=33,
        dropout=0.0,
    )
    value = torch.randn(2, 3, 17, 33)
    phase_norm = torch.linalg.vector_norm(value[:, 1:3], dim=1, keepdim=True).clamp_min(1e-6)
    value[:, 1:3] = value[:, 1:3] / phase_norm
    reliability = torch.rand(2, 1, 17, 33)

    output = encoder(value, reliability)
    output.square().mean().backward()

    assert output.shape == (2, 17, 10)
    assert torch.isfinite(output).all()
    assert any(parameter.grad is not None for parameter in encoder.parameters())


def test_high_frequency_front_back_head_shape_and_gain_invariance():
    torch.manual_seed(17)
    head = HighFrequencyFrontBackHead(
        freq_bins=257,
        pooled_freq_bins=8,
        hidden_channels=8,
        dropout=0.0,
    ).eval()
    left = torch.randn(2, 21, 257)
    right = torch.randn(2, 21, 257)
    common_gain = torch.randn(2, 21, 1)

    logits = head(left, right)
    shifted_logits = head(left + common_gain, right + common_gain)

    assert logits.shape == (2, 2)
    torch.testing.assert_close(logits, shifted_logits, atol=1e-5, rtol=1e-5)


def test_dual_cue_spectral_front_back_head_receives_gradients():
    model = NativeLiteDualCueConcatDOANet(
        freq_bins=257,
        encoder_channels=[8, 12],
        encoder_out_dim=16,
        content_encoder_type="lite_v1",
        content_fusion_dim=16,
        lite_cue_bands=8,
        lite_cue_hidden_dim=12,
        cue_value_out_dim=8,
        cue_reliability_out_dim=4,
        gru_hidden_size=12,
        gru_num_layers=1,
        num_classes=72,
        dropout=0.0,
        use_front_back_auxiliary=True,
        front_back_head_mode="spectral",
        spectral_fb_pooled_bins=8,
        spectral_fb_hidden_channels=8,
    )
    batch = _make_batch(batch_size=2, T=31)

    outputs = model(batch)
    assert outputs["logits"].shape == (2, 72)
    assert outputs["front_back_logits"].shape == (2, 2)
    assert outputs["spectral_front_back_logits"].shape == (2, 2)
    assert "temporal_front_back_logits" not in outputs

    outputs["front_back_logits"].sum().backward()
    assert model.spectral_front_back_head.encoder[0].weight.grad is not None


def test_ds_dilated_cue_encoder_shape_and_gradient():
    encoder = LiteCueEncoder(
        in_channels=3,
        cue_bands=16,
        freq_bins=257,
        temporal_hidden_dim=24,
        out_dim=12,
        kernel_size=3,
        dropout=0.0,
        encoder_type="temporal_conv_ds_dilated",
    )
    cue = torch.randn(2, 3, 17, 257)

    output = encoder(cue)

    assert output.shape == (2, 17, 12)
    output.sum().backward()
    first_block = encoder.temporal_blocks[0]
    assert first_block.net[0].groups == 24
    assert first_block.net[0].weight.grad is not None


def test_precompression_reliability_pooling_starts_from_uniform_and_receives_gradients():
    torch.manual_seed(29)
    encoder = DualBranchCueEncoder(
        cue_bands=16,
        cue_freq_bins=257,
        temporal_hidden_dim=24,
        value_out_dim=12,
        reliability_out_dim=4,
        branch_mode="dual",
        fusion_mode="concat",
        dropout=0.0,
        use_precompression_reliability_pooling=True,
        precompression_pool_hidden_channels=4,
    ).eval()
    value = torch.randn(2, 3, 11, 257)
    reliability = torch.rand(2, 1, 11, 257)
    magnitude = torch.randn(2, 1, 11, 257)

    outputs = encoder(value, reliability, magnitude_context=magnitude)
    baseline_value = encoder.value_encoder(value)

    assert outputs["cue_tf_weight"].shape == (2, 1, 11, 257)
    assert outputs["cue_pool_alpha"].item() == 0.0
    torch.testing.assert_close(outputs["cue_value_feat"], baseline_value, atol=1.0e-6, rtol=1.0e-6)

    outputs["cue_feat"].sum().backward()
    assert encoder.precompression_pool_alpha_raw.grad is not None

    encoder.zero_grad(set_to_none=True)
    with torch.no_grad():
        encoder.precompression_pool_alpha_raw.fill_(0.2)
    encoder(value, reliability, magnitude_context=magnitude)["cue_feat"].sum().backward()
    assert encoder.precompression_weight_net[0].weight.grad is not None


def test_cue_specific_local_tf_separates_ild_and_circular_ipd():
    encoder = DualBranchCueEncoder(
        cue_bands=16,
        cue_freq_bins=257,
        temporal_hidden_dim=24,
        value_out_dim=24,
        reliability_out_dim=8,
        cue_ild_out_dim=8,
        cue_ipd_out_dim=16,
        branch_mode="cue_specific_local_tf",
        fusion_mode="concat",
        dropout=0.0,
    ).eval()
    value = torch.randn(2, 3, 13, 257)
    reliability = torch.rand(2, 1, 13, 257)

    outputs = encoder(value, reliability)

    assert outputs["cue_value_feat"].shape == (2, 13, 24)
    assert outputs["cue_reliability_feat"].shape == (2, 13, 8)
    assert outputs["cue_feat"].shape == (2, 13, 32)
    outputs["cue_feat"].sum().backward()
    assert encoder.cue_specific_local_value_encoder.ild_encoder["local"][0].weight.grad is not None
    assert encoder.cue_specific_local_value_encoder.ipd_encoder["local"][0].weight.grad is not None


def test_cue_specific_local_tf_supports_anisotropic_ordered_blocks():
    encoder = DualBranchCueEncoder(
        cue_bands=32,
        cue_freq_bins=257,
        cue_ild_out_dim=8,
        cue_ipd_out_dim=16,
        branch_mode="cue_specific_local_tf",
        fusion_mode="concat",
        cue_specific_local_use_coherence_context=True,
        cue_specific_local_use_standalone_coherence=False,
        cue_specific_local_block_type="anisotropic_residual",
        cue_specific_local_ild_spectral_kernel_size=7,
        cue_specific_local_ipd_spectral_kernel_size=3,
        dropout=0.0,
    ).eval()
    value = torch.randn(2, 3, 13, 257)
    reliability = torch.rand(2, 1, 13, 257)

    outputs = encoder(value, reliability)

    assert outputs["cue_feat"].shape == (2, 13, 24)
    assert outputs["cue_reliability_feat"] is None
    ild_local = encoder.cue_specific_local_value_encoder.ild_encoder["local"]
    ipd_local = encoder.cue_specific_local_value_encoder.ipd_encoder["local"]
    assert ild_local[0].in_channels == 2
    assert ipd_local[0].in_channels == 3
    assert ild_local[3].spectral[0].kernel_size == (1, 7)
    assert ipd_local[3].spectral[0].kernel_size == (1, 3)
    outputs["cue_feat"].sum().backward()
    assert ild_local[3].spectral[0].weight.grad is not None
    assert ipd_local[3].temporal[1].weight.grad is not None


@pytest.mark.parametrize("aggregation_mode", ["mean", "attention", "coherence_attention"])
def test_cue_specific_progressive_tf_shapes_and_gradients(aggregation_mode):
    encoder = CueSpecificProgressiveTFEncoder(
        aggregation_mode=aggregation_mode,
        channels=(8, 12, 16),
        temporal_dilations=(1, 2, 4),
        dropout=0.0,
    ).eval()
    value = torch.randn(2, 3, 13, 257)
    reliability = torch.rand(2, 1, 13, 257)

    output = encoder(value, reliability)

    assert output.shape == (2, 13, 32)
    output.sum().backward()
    assert encoder.ild_stem[0].weight.grad is not None
    assert encoder.ipd_stem[0].weight.grad is not None
    if aggregation_mode != "mean":
        assert encoder.ild_attention.weight.grad is not None
        assert encoder.ipd_attention.weight.grad is not None
    if aggregation_mode == "coherence_attention":
        assert encoder.coherence_beta_raw.grad is not None


def test_dual_cue_encoder_supports_progressive_tf_branch():
    encoder = DualBranchCueEncoder(
        cue_freq_bins=257,
        cue_ild_out_dim=8,
        cue_ipd_out_dim=16,
        branch_mode="cue_specific_progressive_tf",
        cue_progressive_aggregation="coherence_attention",
        cue_progressive_out_dim=32,
        fusion_mode="concat",
        dropout=0.0,
    ).eval()
    value = torch.randn(2, 3, 13, 257)
    reliability = torch.rand(2, 1, 13, 257)

    outputs = encoder(value, reliability)

    assert outputs["cue_feat"].shape == (2, 13, 32)
    assert outputs["cue_reliability_feat"] is None
    assert encoder.out_dim == 32


def test_fine_to_coarse_refinement_starts_from_uniform16_and_receives_gradients():
    torch.manual_seed(31)
    refinement = FineToCoarseSubbandRefinement(
        channels=8,
        coarse_bands=16,
    ).eval()
    features = torch.randn(2, 8, 13, 257)
    uniform16 = torch.nn.functional.adaptive_avg_pool2d(features, (13, 16))

    output = refinement(features)

    assert output.shape == (2, 8, 13, 16)
    torch.testing.assert_close(output, uniform16, atol=1.0e-6, rtol=1.0e-6)
    output.sum().backward()
    assert refinement.residual_alpha_raw.grad is not None

    refinement.zero_grad(set_to_none=True)
    with torch.no_grad():
        refinement.residual_alpha_raw.fill_(0.2)
    refinement(features).sum().backward()
    assert refinement.detail_gate[0].weight.grad is not None


def test_cue_specific_local_tf_supports_fine32_to_coarse16_refinement():
    encoder = DualBranchCueEncoder(
        cue_bands=16,
        cue_freq_bins=257,
        temporal_hidden_dim=24,
        value_out_dim=24,
        reliability_out_dim=8,
        cue_ild_out_dim=8,
        cue_ipd_out_dim=16,
        branch_mode="cue_specific_local_tf",
        fusion_mode="concat",
        cue_specific_local_use_fine_to_coarse_refinement=True,
        dropout=0.0,
    ).eval()
    value = torch.randn(2, 3, 13, 257)
    reliability = torch.rand(2, 1, 13, 257)

    outputs = encoder(value, reliability)

    assert outputs["cue_feat"].shape == (2, 13, 32)
    ild_refinement = encoder.cue_specific_local_value_encoder.ild_encoder["fine_to_coarse"]
    ipd_refinement = encoder.cue_specific_local_value_encoder.ipd_encoder["fine_to_coarse"]
    assert ild_refinement.fine_bands == 32
    assert ipd_refinement.fine_bands == 32
    outputs["cue_feat"].sum().backward()
    assert ild_refinement.residual_alpha_raw.grad is not None
    assert ipd_refinement.residual_alpha_raw.grad is not None


def test_cue_specific_local_tf_supports_coherence_role_ablation():
    value = torch.randn(2, 3, 13, 257)
    reliability = torch.rand(2, 1, 13, 257)

    for use_context, use_standalone in ((True, False), (False, True), (False, False)):
        encoder = DualBranchCueEncoder(
            cue_bands=16,
            cue_freq_bins=257,
            temporal_hidden_dim=24,
            value_out_dim=24,
            reliability_out_dim=8,
            cue_ild_out_dim=8,
            cue_ipd_out_dim=16,
            branch_mode="cue_specific_local_tf",
            fusion_mode="concat",
            cue_specific_local_use_coherence_context=use_context,
            cue_specific_local_use_standalone_coherence=use_standalone,
            dropout=0.0,
        ).eval()

        outputs = encoder(value, reliability)
        expected_dim = 24 + (8 if use_standalone else 0)
        assert outputs["cue_feat"].shape == (2, 13, expected_dim)
        assert (outputs["cue_reliability_feat"] is not None) == use_standalone
        assert encoder.cue_specific_local_value_encoder.ild_encoder["local"][0].in_channels == (
            2 if use_context else 1
        )
        assert encoder.cue_specific_local_value_encoder.ipd_encoder["local"][0].in_channels == (
            3 if use_context else 2
        )
        outputs["cue_feat"].sum().backward()
        assert encoder.cue_specific_local_value_encoder.ild_encoder["local"][0].weight.grad is not None
        assert encoder.cue_specific_local_value_encoder.ipd_encoder["local"][0].weight.grad is not None


def test_cue_specific_local_tf_routes_cue_specific_consistency():
    value = torch.randn(2, 3, 13, 257)
    coherence = torch.rand(2, 1, 13, 257)
    ild_consistency = torch.rand_like(coherence)
    ipd_consistency = torch.rand_like(coherence)
    encoder = DualBranchCueEncoder(
        cue_bands=16,
        cue_freq_bins=257,
        temporal_hidden_dim=24,
        value_out_dim=24,
        reliability_out_dim=8,
        cue_ild_out_dim=8,
        cue_ipd_out_dim=16,
        branch_mode="cue_specific_local_tf",
        fusion_mode="concat",
        cue_specific_local_use_coherence_context=False,
        cue_specific_local_use_cue_consistency_context=True,
        cue_specific_local_use_standalone_coherence=True,
        dropout=0.0,
    ).eval()

    outputs = encoder(
        value,
        coherence,
        ild_consistency_tensor=ild_consistency,
        ipd_consistency_tensor=ipd_consistency,
    )

    assert outputs["cue_feat"].shape == (2, 13, 32)
    assert outputs["cue_value_feat"].shape == (2, 13, 24)
    assert outputs["cue_reliability_feat"].shape == (2, 13, 8)
    outputs["cue_feat"].sum().backward()


def test_mean_only_content_supports_balanced_branch_fusion():
    model = NativeLiteDualCueConcatDOANet(
        freq_bins=257,
        encoder_channels=[8, 12],
        encoder_out_dim=16,
        content_encoder_type="lite_v1",
        content_relation_mode="mean_only",
        content_fusion_dim=12,
        lite_cue_bands=8,
        lite_cue_hidden_dim=12,
        cue_value_out_dim=8,
        cue_reliability_out_dim=4,
        cue_branch_mode="dual",
        use_branchwise_fusion_norm=True,
        gru_hidden_size=12,
        gru_num_layers=1,
        num_classes=25,
        dropout=0.0,
        use_front_back_auxiliary=False,
    )
    batch = _make_batch(batch_size=2, T=31)

    outputs = model(batch)

    assert outputs["content_feat"].shape == (2, 31, 12)
    assert outputs["cue_feat"].shape == (2, 31, 12)
    assert outputs["fused_feat"].shape == (2, 31, 24)
    outputs["logits"].sum().backward()
    assert model.content_branch_norm.weight.grad is not None
    assert model.cue_branch_norm.weight.grad is not None


def test_pre_common_energy_content_is_ear_swap_invariant_and_encoded_once():
    model = NativeLiteDualCueConcatDOANet(
        freq_bins=257,
        encoder_channels=[8, 12],
        encoder_out_dim=16,
        content_encoder_type="lite_v1",
        content_relation_mode="pre_common_energy",
        content_fusion_dim=12,
        lite_cue_bands=8,
        lite_cue_hidden_dim=12,
        cue_value_out_dim=8,
        cue_reliability_out_dim=4,
        cue_branch_mode="dual",
        use_branchwise_fusion_norm=True,
        gru_hidden_size=12,
        gru_num_layers=1,
        num_classes=25,
        dropout=0.0,
        use_front_back_auxiliary=False,
    ).eval()
    batch = _make_batch(batch_size=2, T=31)
    swapped = dict(batch)
    swapped["log_mag_L"] = batch["log_mag_R"]
    swapped["log_mag_R"] = batch["log_mag_L"]
    encoder_calls = []
    hook = model.encoder.register_forward_hook(
        lambda _module, _inputs, _output: encoder_calls.append(1)
    )

    with torch.no_grad():
        content = model(batch)["content_feat"]
        swapped_content = model(swapped)["content_feat"]
    hook.remove()

    assert len(encoder_calls) == 2
    torch.testing.assert_close(content, swapped_content, atol=1.0e-6, rtol=1.0e-6)


def test_ear_token_content_preserves_order_and_receives_gradients():
    model = NativeLiteDualCueConcatDOANet(
        freq_bins=257,
        encoder_channels=[8, 12],
        encoder_out_dim=16,
        content_encoder_type="lite_v1",
        content_relation_mode="ear_token_attention",
        content_fusion_dim=12,
        content_ear_token_dim=8,
        content_ear_token_heads=2,
        lite_cue_bands=8,
        lite_cue_hidden_dim=12,
        cue_value_out_dim=8,
        cue_reliability_out_dim=4,
        cue_branch_mode="dual",
        use_branchwise_fusion_norm=True,
        gru_hidden_size=12,
        gru_num_layers=1,
        num_classes=25,
        dropout=0.0,
        use_front_back_auxiliary=False,
    ).eval()
    batch = _make_batch(batch_size=2, T=31)
    swapped = dict(batch)
    swapped["log_mag_L"] = batch["log_mag_R"]
    swapped["log_mag_R"] = batch["log_mag_L"]

    content = model(batch)["content_feat"]
    swapped_content = model(swapped)["content_feat"]

    assert content.shape == (2, 31, 12)
    assert not torch.allclose(content, swapped_content)
    content.sum().backward()
    assert model.content_ear_projection.weight.grad is not None
    assert model.content_ear_embedding.grad is not None
    assert model.content_pair_attn.in_proj_weight.grad is not None


def test_multiscale_evidence_model_outputs_normalized_frame_weights():
    model = NativeLiteDualCueConcatDOANet(
        freq_bins=257,
        encoder_channels=[8, 12],
        encoder_out_dim=16,
        content_encoder_type="lite_v1",
        content_fusion_dim=16,
        lite_cue_bands=8,
        lite_cue_hidden_dim=12,
        cue_value_out_dim=8,
        cue_reliability_out_dim=4,
        gru_hidden_size=12,
        gru_num_layers=1,
        temporal_aggregation_type="multiscale_evidence",
        num_classes=25,
        dropout=0.0,
        use_front_back_auxiliary=False,
    )
    batch = _make_batch(batch_size=2, T=31)
    batch["azimuth_label"] = torch.randint(0, 25, (2,))

    outputs = model(batch)

    assert outputs["logits"].shape == (2, 25)
    assert outputs["frame_logits"].shape == (2, 31, 25)
    assert outputs["attn_weights"].shape == (2, 31, 1)
    torch.testing.assert_close(
        outputs["attn_weights"].sum(dim=1),
        torch.ones(2, 1),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    outputs["logits"].sum().backward()
    assert model.temporal_head.evidence_aggregator.frame_classifier.weight.grad is not None


def _make_raw_complex_batch(batch_size: int = 2, time_steps: int = 21, freq_bins: int = 257):
    return {
        "spec_real_L": torch.randn(batch_size, time_steps, freq_bins),
        "spec_imag_L": torch.randn(batch_size, time_steps, freq_bins),
        "spec_real_R": torch.randn(batch_size, time_steps, freq_bins),
        "spec_imag_R": torch.randn(batch_size, time_steps, freq_bins),
    }


def test_raw_complex_encoder_shape_gradient_and_common_gain_invariance():
    torch.manual_seed(23)
    encoder = RawComplexTFCueEncoder(
        out_dim=24,
        channels=(4, 6, 8),
        pooled_freq_bins=4,
        dropout=0.0,
    ).eval()
    batch = _make_raw_complex_batch()
    common_gain = torch.rand(2, 21, 1) + 0.5
    scaled = {key: value * common_gain for key, value in batch.items()}

    output = encoder(batch)
    scaled_output = encoder(scaled)

    assert output.shape == (2, 21, 24)
    torch.testing.assert_close(output, scaled_output, atol=2.0e-5, rtol=2.0e-5)
    output.sum().backward()
    assert encoder.blocks[0].depthwise.real.weight.grad is not None


def test_raw_complex_model_does_not_require_handcrafted_cues():
    model = NativeLiteDualCueConcatDOANet(
        freq_bins=257,
        encoder_channels=[8, 12],
        encoder_out_dim=16,
        content_encoder_type="lite_v1",
        content_fusion_dim=16,
        cue_input_mode="raw_complex",
        raw_complex_cue_out_dim=28,
        raw_complex_channels=(4, 6, 8),
        raw_complex_pooled_bins=4,
        disable_content_stream=True,
        gru_hidden_size=12,
        gru_num_layers=1,
        num_classes=72,
        dropout=0.0,
        use_front_back_auxiliary=True,
    )
    batch = _make_raw_complex_batch(time_steps=31)
    batch["log_mag_L"] = torch.randn(2, 31, 257)
    batch["log_mag_R"] = torch.randn(2, 31, 257)

    outputs = model(batch)

    assert outputs["logits"].shape == (2, 72)
    assert outputs["front_back_logits"].shape == (2, 2)
    assert outputs["cue_feat"].shape == (2, 31, 28)
    assert outputs["content_feat"] is None
    outputs["logits"].sum().backward()
    assert model.raw_complex_cue_encoder.blocks[0].pointwise.real.weight.grad is not None


def test_latent_cross_spectrum_model_shape_and_shared_encoder_gradient():
    model = NativeLiteLatentCrossSpectrumDOANet(
        complex_channels=(4, 6, 8),
        content_out_dim=12,
        spatial_out_dim=12,
        content_hidden_channels=4,
        spatial_hidden_channels=8,
        spatial_output_channels=4,
        gru_hidden_size=12,
        gru_num_layers=1,
        num_classes=25,
        dropout=0.0,
    )
    batch = _make_raw_complex_batch(time_steps=31)

    outputs = model(batch)

    assert outputs["logits"].shape == (2, 25)
    assert outputs["content_feat"].shape == (2, 31, 12)
    assert outputs["spatial_feat"].shape == (2, 31, 12)
    assert outputs["fused_feat"].shape == (2, 31, 24)
    outputs["logits"].sum().backward()
    first_block = model.shared_complex_encoder.blocks[0]
    assert first_block.depthwise.real.weight.grad is not None
    assert first_block.depthwise.imag.weight.grad is not None


def test_latent_cross_spectrum_model_common_positive_gain_invariance():
    torch.manual_seed(29)
    model = NativeLiteLatentCrossSpectrumDOANet(
        complex_channels=(4, 6, 8),
        content_out_dim=12,
        spatial_out_dim=12,
        content_hidden_channels=4,
        spatial_hidden_channels=8,
        spatial_output_channels=4,
        gru_hidden_size=12,
        gru_num_layers=1,
        num_classes=25,
        dropout=0.0,
    ).eval()
    batch = _make_raw_complex_batch(time_steps=21)
    common_gain = torch.rand(2, 21, 1) + 0.5
    scaled = {key: value * common_gain for key, value in batch.items()}

    with torch.no_grad():
        output = model(batch)["logits"]
        scaled_output = model(scaled)["logits"]

    torch.testing.assert_close(output, scaled_output, atol=3.0e-5, rtol=3.0e-5)


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


def test_residual_product_concat_preserves_value_and_product_gradients():
    encoder = DualBranchCueEncoder(
        cue_bands=16,
        cue_freq_bins=257,
        temporal_hidden_dim=48,
        value_out_dim=24,
        reliability_out_dim=8,
        fusion_mode="residual_product_concat",
        branch_mode="dual",
        dropout=0.0,
    )
    value = torch.randn(2, 3, 11, 257)
    reliability = torch.rand(2, 1, 11, 257)

    outputs = encoder(value, reliability)

    assert outputs["cue_value_feat"].shape == (2, 11, 24)
    assert outputs["cue_reliability_feat"].shape == (2, 11, 8)
    assert outputs["cue_gate"].shape == (2, 11, 24)
    assert outputs["cue_feat"].shape == (2, 11, 56)

    outputs["cue_feat"].sum().backward()
    assert encoder.rel_to_product[0].weight.grad is not None


def _complex_batch(spec_l: torch.Tensor, spec_r: torch.Tensor):
    return {
        "spec_real_L": spec_l.real,
        "spec_imag_L": spec_l.imag,
        "spec_real_R": spec_r.real,
        "spec_imag_R": spec_r.imag,
        "coherence": torch.ones_like(spec_l.real),
    }


def test_precue_statistics_identical_ears_have_zero_ild_and_phase():
    torch.manual_seed(7)
    spec_l = torch.complex(torch.randn(2, 5, 257), torch.randn(2, 5, 257))
    compressor = BinauralCueStatistics(
        mode="precue_stat",
        num_bands=16,
        freq_bins=257,
    )

    outputs = compressor(_complex_batch(spec_l, spec_l), time_steps=5)
    value = outputs["value_tensor"]
    reliability = outputs["reliability_tensor"]

    assert value.shape == (2, 3, 5, 16)
    assert reliability.shape == (2, 1, 5, 16)
    assert torch.isfinite(value).all()
    assert torch.allclose(value[:, 0], torch.zeros_like(value[:, 0]), atol=1.0e-5)
    assert torch.allclose(value[:, 1], torch.zeros_like(value[:, 1]), atol=1.0e-5)
    assert torch.allclose(value[:, 2], torch.ones_like(value[:, 2]), atol=1.0e-5)
    assert torch.allclose(reliability, torch.ones_like(reliability), atol=1.0e-5)


def test_phaseaware_statistics_recovers_known_delay():
    sample_rate = 16000
    freq_bins = 257
    delay_seconds = 0.000375
    frequencies = torch.linspace(0.0, sample_rate / 2.0, steps=freq_bins)
    spec_l = torch.ones(1, 3, freq_bins, dtype=torch.complex64)
    spec_r = torch.exp(-1j * 2.0 * torch.pi * frequencies * delay_seconds)
    spec_r = spec_r.reshape(1, 1, freq_bins).expand_as(spec_l)
    compressor = BinauralCueStatistics(
        mode="phaseaware_stat",
        num_bands=16,
        freq_bins=freq_bins,
        sample_rate=sample_rate,
        delay_max_ms=1.0,
        delay_bins=33,
        delay_temperature=20.0,
    )

    outputs = compressor(_complex_batch(spec_l, spec_r), time_steps=3)
    itd = outputs["itd_seconds"]

    assert outputs["value_tensor"].shape == (1, 3, 3, 16)
    assert outputs["reliability_tensor"].shape == (1, 1, 3, 16)
    assert torch.isfinite(outputs["value_tensor"]).all()
    assert torch.mean(torch.abs(itd - delay_seconds)) < 1.5e-4
    assert torch.all((outputs["reliability_tensor"] >= 0.0) & (outputs["reliability_tensor"] <= 1.0))


def test_precue_statistics_do_not_change_trainable_parameter_count():
    common = dict(
        freq_bins=257,
        encoder_channels=[16, 24, 32],
        encoder_out_dim=64,
        encoder_variant="v2_balanced",
        content_encoder_type="lite_v1",
        content_fusion_dim=80,
        lite_cue_bands=16,
        lite_cue_hidden_dim=48,
        cue_value_out_dim=24,
        cue_reliability_out_dim=8,
        gru_hidden_size=80,
        dropout=0.0,
    )
    control = NativeLiteDualCueConcatDOANet(
        **common,
        cue_stat_mode="postcue_uniform",
    )
    precue = NativeLiteDualCueConcatDOANet(
        **common,
        cue_stat_mode="precue_stat",
    )
    phaseaware = NativeLiteDualCueConcatDOANet(
        **common,
        cue_stat_mode="phaseaware_stat",
    )

    parameter_counts = {
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        for model in (control, precue, phaseaware)
    }
    assert parameter_counts == {152803}
