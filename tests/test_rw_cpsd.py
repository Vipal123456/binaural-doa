import torch

from models.native_lite_v7 import (
    CalibratedPrecisionCueFactorizedCPSD,
    CueFactorizedCPSD,
    NonlinearCueFactorizedCPSD,
    NonlinearOracleSupervisedCueFactorizedCPSD,
    OracleSupervisedCueFactorizedCPSD,
    OracleTargetCPSD,
    OracleTargetMaskedCPSD,
    PrecisionWeightedCueFactorizedCPSD,
    ReliabilityWeightedCPSD,
    TargetAwareCueFactorizedCPSD,
    TargetAwareRWCPSD,
)


def _batch(spec_l: torch.Tensor, spec_r: torch.Tensor):
    return {
        "spec_real_L": spec_l.real,
        "spec_imag_L": spec_l.imag,
        "spec_real_R": spec_r.real,
        "spec_imag_R": spec_r.imag,
    }


def test_rw_cpsd_starts_as_uniform_five_frame_estimator():
    torch.manual_seed(0)
    spec_l = torch.randn(2, 11, 7, dtype=torch.complex64)
    spec_r = torch.randn(2, 11, 7, dtype=torch.complex64)
    module = ReliabilityWeightedCPSD(time_frames=5)

    output = module(_batch(spec_l, spec_r), time_steps=11)
    weights = output["tf_weight"]

    assert weights.shape == (2, 11, 7, 5)
    torch.testing.assert_close(weights, torch.full_like(weights, 0.2))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
    assert output["reliability_tensor"].amax() <= 1.0
    assert output["reliability_tensor"].amin() >= 0.0
    assert output["ild_consistency_tensor"].shape == (2, 1, 11, 7)
    assert output["ipd_consistency_tensor"].shape == (2, 1, 11, 7)
    assert output["ild_consistency_tensor"].amin() >= 0.0
    assert output["ild_consistency_tensor"].amax() <= 1.0
    assert output["ipd_consistency_tensor"].amin() >= 0.0
    assert output["ipd_consistency_tensor"].amax() <= 1.0


def test_rw_cpsd_consistency_distinguishes_level_and_phase_variation():
    phase = torch.zeros(1, 7, 1)
    spec_l = torch.ones(1, 7, 1, dtype=torch.complex64)
    spec_r_level = torch.polar(
        torch.tensor([1.0, 0.25, 1.0, 4.0, 1.0, 0.25, 1.0]).view(1, 7, 1),
        phase,
    )
    spec_r_phase = torch.polar(
        torch.ones(1, 7, 1),
        torch.tensor([0.0, 1.2, 0.0, -1.2, 0.0, 1.2, 0.0]).view(1, 7, 1),
    )
    module = ReliabilityWeightedCPSD(time_frames=5)

    level_output = module(_batch(spec_l, spec_r_level), time_steps=7)
    phase_output = module(_batch(spec_l, spec_r_phase), time_steps=7)

    center = 3
    assert (
        level_output["ild_consistency_tensor"][0, 0, center, 0]
        < phase_output["ild_consistency_tensor"][0, 0, center, 0]
    )
    assert (
        phase_output["ipd_consistency_tensor"][0, 0, center, 0]
        < level_output["ipd_consistency_tensor"][0, 0, center, 0]
    )


def test_rw_cpsd_score_coefficients_receive_gradients():
    torch.manual_seed(1)
    spec_l = torch.randn(2, 9, 5, dtype=torch.complex64)
    spec_r = torch.randn(2, 9, 5, dtype=torch.complex64)
    module = ReliabilityWeightedCPSD(time_frames=5)

    output = module(_batch(spec_l, spec_r), time_steps=9)
    loss = output["value_tensor"].square().mean()
    loss.backward()

    gradient = module.score_coefficients.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_frequency_aware_rw_cpsd_starts_equivalent_to_global_mode():
    torch.manual_seed(2)
    spec_l = torch.randn(2, 9, 17, dtype=torch.complex64)
    spec_r = torch.randn(2, 9, 17, dtype=torch.complex64)
    batch = _batch(spec_l, spec_r)
    global_module = ReliabilityWeightedCPSD(time_frames=5)
    frequency_module = ReliabilityWeightedCPSD(
        time_frames=5,
        coefficient_mode="frequency_anchors",
        frequency_anchors=8,
    )

    global_output = global_module(batch, time_steps=9)
    frequency_output = frequency_module(batch, time_steps=9)

    torch.testing.assert_close(
        frequency_output["tf_weight"], global_output["tf_weight"]
    )
    torch.testing.assert_close(
        frequency_output["value_tensor"], global_output["value_tensor"]
    )


def test_frequency_aware_rw_cpsd_interpolates_coefficients_and_backpropagates():
    torch.manual_seed(3)
    module = ReliabilityWeightedCPSD(
        time_frames=5,
        coefficient_mode="frequency_anchors",
        frequency_anchors=4,
    )
    with torch.no_grad():
        module.score_coefficients[0].copy_(torch.tensor([-1.0, -0.5, 0.5, 1.0]))

    coefficients = module.frequency_score_coefficients(13)
    assert coefficients.shape == (3, 13)
    torch.testing.assert_close(coefficients[:, 0], module.score_coefficients[:, 0])
    torch.testing.assert_close(coefficients[:, -1], module.score_coefficients[:, -1])
    assert not torch.equal(coefficients[0, 0], coefficients[0, -1])

    spec_l = torch.randn(2, 9, 13, dtype=torch.complex64)
    spec_r = torch.randn(2, 9, 13, dtype=torch.complex64)
    output = module(_batch(spec_l, spec_r), time_steps=9)
    output["value_tensor"].square().mean().backward()

    gradient = module.score_coefficients.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert (gradient.abs().sum(dim=0) > 0).all()


def test_cue_factorized_cpsd_starts_as_uniform_estimator():
    torch.manual_seed(4)
    spec_l = torch.randn(2, 11, 7, dtype=torch.complex64)
    spec_r = torch.randn(2, 11, 7, dtype=torch.complex64)
    module = CueFactorizedCPSD(time_frames=5)

    output = module(_batch(spec_l, spec_r), time_steps=11)
    expected = torch.full_like(output["tf_weight_ild"], 0.2)

    torch.testing.assert_close(output["tf_weight_ild"], expected)
    torch.testing.assert_close(output["tf_weight_ipd"], expected)
    torch.testing.assert_close(output["tf_weight"], expected)


def test_cue_factorized_cpsd_has_independent_trainable_weights():
    torch.manual_seed(5)
    spec_l = torch.randn(2, 9, 5, dtype=torch.complex64)
    spec_r = torch.randn(2, 9, 5, dtype=torch.complex64)
    module = CueFactorizedCPSD(time_frames=5)

    with torch.no_grad():
        module.ild_score_coefficients.copy_(torch.tensor([0.2, 0.7]))
        module.ipd_score_coefficients.copy_(torch.tensor([-0.3, 0.5]))
    output = module(_batch(spec_l, spec_r), time_steps=9)

    assert not torch.equal(output["tf_weight_ild"], output["tf_weight_ipd"])
    loss = output["value_tensor"].square().mean()
    loss.backward()
    for gradient in (
        module.ild_score_coefficients.grad,
        module.ipd_score_coefficients.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_nonlinear_cue_factorized_cpsd_starts_as_b2():
    torch.manual_seed(10)
    spec_l = torch.randn(2, 9, 7, dtype=torch.complex64)
    spec_r = torch.randn(2, 9, 7, dtype=torch.complex64)
    batch = _batch(spec_l, spec_r)
    baseline = CueFactorizedCPSD(time_frames=5)
    nonlinear = NonlinearCueFactorizedCPSD(time_frames=5)

    baseline_output = baseline(batch, time_steps=9)
    nonlinear_output = nonlinear(batch, time_steps=9)

    torch.testing.assert_close(
        nonlinear_output["tf_weight_ild"], baseline_output["tf_weight_ild"]
    )
    torch.testing.assert_close(
        nonlinear_output["tf_weight_ipd"], baseline_output["tf_weight_ipd"]
    )
    torch.testing.assert_close(
        nonlinear_output["value_tensor"], baseline_output["value_tensor"]
    )


def test_nonlinear_cue_factorized_heads_receive_gradients():
    torch.manual_seed(11)
    spec_l = torch.randn(2, 9, 7, dtype=torch.complex64)
    spec_r = torch.randn(2, 9, 7, dtype=torch.complex64)
    module = NonlinearCueFactorizedCPSD(time_frames=5)

    output = module(_batch(spec_l, spec_r), time_steps=9)
    output["value_tensor"].square().mean().backward()

    for head in (module.ild_score_residual, module.ipd_score_residual):
        gradient = head[-1].weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_oracle_target_cpsd_uses_target_spectra():
    mixture_l = torch.randn(1, 9, 3, dtype=torch.complex64)
    mixture_r = torch.randn(1, 9, 3, dtype=torch.complex64)
    target_l = torch.ones(1, 9, 3, dtype=torch.complex64)
    target_r = torch.ones(1, 9, 3, dtype=torch.complex64)
    batch = _batch(mixture_l, mixture_r)
    batch.update({f"target_{key}": value for key, value in _batch(target_l, target_r).items()})
    output = OracleTargetCPSD(time_frames=5)(batch, time_steps=9)

    torch.testing.assert_close(output["value_tensor"][:, 0], torch.zeros(1, 9, 3))
    torch.testing.assert_close(output["value_tensor"][:, 1], torch.zeros(1, 9, 3))
    torch.testing.assert_close(output["value_tensor"][:, 2], torch.ones(1, 9, 3))


def test_oracle_target_mask_prefers_target_dominant_frame():
    mixture_l = torch.ones(1, 7, 1, dtype=torch.complex64)
    mixture_r = torch.ones(1, 7, 1, dtype=torch.complex64)
    target_level = torch.tensor([0.1, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1]).view(1, 7, 1)
    target_l = torch.complex(target_level, torch.zeros_like(target_level))
    target_r = target_l.clone()
    interferer = torch.ones(1, 7, 1, dtype=torch.complex64)
    batch = _batch(mixture_l, mixture_r)
    batch.update({f"target_{key}": value for key, value in _batch(target_l, target_r).items()})
    batch.update({f"interferer_{key}": value for key, value in _batch(interferer, interferer).items()})
    output = OracleTargetMaskedCPSD(time_frames=5)(batch, time_steps=7)

    center_weights = output["tf_weight"][0, 3, 0]
    assert center_weights.argmax().item() == 2


def test_target_aware_cue_factorized_cpsd_produces_auxiliary_losses():
    torch.manual_seed(6)
    mixture_l = torch.randn(2, 9, 5, dtype=torch.complex64)
    mixture_r = torch.randn(2, 9, 5, dtype=torch.complex64)
    target_l = 0.7 * torch.randn(2, 9, 5, dtype=torch.complex64)
    target_r = 0.7 * torch.randn(2, 9, 5, dtype=torch.complex64)
    interferer_l = 0.3 * torch.randn(2, 9, 5, dtype=torch.complex64)
    interferer_r = 0.3 * torch.randn(2, 9, 5, dtype=torch.complex64)
    batch = _batch(mixture_l, mixture_r)
    batch.update({f"target_{key}": value for key, value in _batch(target_l, target_r).items()})
    batch.update({
        f"interferer_{key}": value
        for key, value in _batch(interferer_l, interferer_r).items()
    })
    module = TargetAwareCueFactorizedCPSD(time_frames=5)

    output = module(batch, time_steps=9)
    assert output["target_probability"].shape == (2, 9, 5)
    assert torch.isfinite(output["target_mask_loss"])
    assert torch.isfinite(output["target_covariance_loss"])
    loss = (
        output["value_tensor"].square().mean()
        + output["target_mask_loss"]
        + output["target_covariance_loss"]
    )
    loss.backward()
    gradient = module.target_dominance_head.net[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_target_cue_residual_starts_as_plain_cue_factorized_cpsd():
    torch.manual_seed(8)
    mixture_l = torch.randn(2, 9, 5, dtype=torch.complex64)
    mixture_r = torch.randn(2, 9, 5, dtype=torch.complex64)
    batch = _batch(mixture_l, mixture_r)
    baseline = CueFactorizedCPSD(time_frames=5)
    residual = TargetAwareCueFactorizedCPSD(
        time_frames=5,
        target_bias_mode="cue_residual",
    )

    baseline_output = baseline(batch, time_steps=9)
    residual_output = residual(batch, time_steps=9)

    torch.testing.assert_close(
        residual_output["tf_weight_ild"], baseline_output["tf_weight_ild"]
    )
    torch.testing.assert_close(
        residual_output["tf_weight_ipd"], baseline_output["tf_weight_ipd"]
    )
    torch.testing.assert_close(
        residual_output["value_tensor"], baseline_output["value_tensor"]
    )
    torch.testing.assert_close(
        residual.target_bias_coefficients, torch.zeros(2)
    )


def test_oracle_supervised_cue_factorized_cpsd_has_finite_weight_loss():
    torch.manual_seed(9)
    target_l = torch.randn(2, 9, 5, dtype=torch.complex64)
    target_r = torch.randn(2, 9, 5, dtype=torch.complex64)
    interferer_l = 0.5 * torch.randn(2, 9, 5, dtype=torch.complex64)
    interferer_r = 0.5 * torch.randn(2, 9, 5, dtype=torch.complex64)
    batch = _batch(target_l + interferer_l, target_r + interferer_r)
    batch.update({f"target_{key}": value for key, value in _batch(target_l, target_r).items()})
    module = OracleSupervisedCueFactorizedCPSD(time_frames=5)

    output = module(batch, time_steps=9)

    assert torch.isfinite(output["cue_reliability_loss"])
    torch.testing.assert_close(
        output["oracle_tf_weight_ild"].sum(dim=-1),
        torch.ones_like(output["oracle_tf_weight_ild"][..., 0]),
    )
    torch.testing.assert_close(
        output["oracle_tf_weight_ipd"].sum(dim=-1),
        torch.ones_like(output["oracle_tf_weight_ipd"][..., 0]),
    )
    output["cue_reliability_loss"].backward()
    for gradient in (
        module.ild_score_coefficients.grad,
        module.ipd_score_coefficients.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_nonlinear_oracle_supervision_updates_residual_heads():
    torch.manual_seed(12)
    target_l = torch.randn(2, 9, 5, dtype=torch.complex64)
    target_r = torch.randn(2, 9, 5, dtype=torch.complex64)
    interferer_l = 0.5 * torch.randn(2, 9, 5, dtype=torch.complex64)
    interferer_r = 0.5 * torch.randn(2, 9, 5, dtype=torch.complex64)
    batch = _batch(target_l + interferer_l, target_r + interferer_r)
    batch.update(
        {f"target_{key}": value for key, value in _batch(target_l, target_r).items()}
    )
    module = NonlinearOracleSupervisedCueFactorizedCPSD(time_frames=5)

    output = module(batch, time_steps=9)
    assert torch.isfinite(output["cue_reliability_loss"])
    output["cue_reliability_loss"].backward()

    for head in (module.ild_score_residual, module.ipd_score_residual):
        gradient = head[-1].weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_precision_weighted_cpsd_starts_uniform_and_receives_gradients():
    torch.manual_seed(13)
    spec_l = torch.randn(2, 9, 5, dtype=torch.complex64)
    spec_r = torch.randn(2, 9, 5, dtype=torch.complex64)
    module = PrecisionWeightedCueFactorizedCPSD(time_frames=5)

    output = module(_batch(spec_l, spec_r), time_steps=9)

    torch.testing.assert_close(
        output["tf_weight_ild"],
        torch.full_like(output["tf_weight_ild"], 0.2),
    )
    torch.testing.assert_close(
        output["tf_weight_ipd"],
        torch.full_like(output["tf_weight_ipd"], 0.2),
    )
    assert output["cue_uncertainty_loss"] is None
    output["value_tensor"].square().mean().backward()
    for head in (
        module.ild_log_variance_head,
        module.ipd_log_concentration_head,
    ):
        gradient = head[-1].weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_calibrated_precision_cpsd_has_finite_aggregate_nll():
    torch.manual_seed(14)
    target_l = torch.randn(2, 9, 5, dtype=torch.complex64)
    target_r = torch.randn(2, 9, 5, dtype=torch.complex64)
    interferer_l = 0.5 * torch.randn(2, 9, 5, dtype=torch.complex64)
    interferer_r = 0.5 * torch.randn(2, 9, 5, dtype=torch.complex64)
    batch = _batch(target_l + interferer_l, target_r + interferer_r)
    batch.update(
        {f"target_{key}": value for key, value in _batch(target_l, target_r).items()}
    )
    module = CalibratedPrecisionCueFactorizedCPSD(time_frames=5)

    output = module(batch, time_steps=9)

    assert torch.isfinite(output["cue_uncertainty_loss"])
    assert torch.isfinite(output["cue_uncertainty_ild_loss"])
    assert torch.isfinite(output["cue_uncertainty_ipd_loss"])
    assert output["aggregate_ild_log_variance"].shape == (2, 9, 5)
    assert output["aggregate_ipd_concentration"].shape == (2, 9, 5)
    output["cue_uncertainty_loss"].backward()
    for head in (
        module.ild_log_variance_head,
        module.ipd_log_concentration_head,
    ):
        gradient = head[-1].weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_target_aware_rw_cpsd_produces_mask_loss():
    torch.manual_seed(7)
    target_l = torch.randn(2, 9, 5, dtype=torch.complex64)
    target_r = torch.randn(2, 9, 5, dtype=torch.complex64)
    interferer_l = 0.5 * torch.randn(2, 9, 5, dtype=torch.complex64)
    interferer_r = 0.5 * torch.randn(2, 9, 5, dtype=torch.complex64)
    batch = _batch(target_l + interferer_l, target_r + interferer_r)
    batch.update({f"target_{key}": value for key, value in _batch(target_l, target_r).items()})
    batch.update({
        f"interferer_{key}": value
        for key, value in _batch(interferer_l, interferer_r).items()
    })
    module = TargetAwareRWCPSD(time_frames=5)
    output = module(batch, time_steps=9)

    assert output["target_probability"].shape == (2, 9, 5)
    assert torch.isfinite(output["target_mask_loss"])
    output["target_mask_loss"].backward()
    gradient = module.target_dominance_head.net[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
