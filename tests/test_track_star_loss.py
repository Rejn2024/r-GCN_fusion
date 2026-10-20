"""Compare optimized HGT losses to independently grouped Dirichlet formulas."""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from rgcn_fusion.track_star_loss import (
    batch_hierarchical_evidential_loss,
    component_expected_dirichlet_ce,
)


def make_case(counts, classes, dtype=torch.float32, device="cpu"):
    generator = torch.Generator().manual_seed(139)
    counts = torch.tensor(counts, dtype=torch.long)
    membership = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    membership = membership[torch.randperm(membership.numel(), generator=generator)]
    modes = 6
    component_ids = torch.stack([torch.arange(classes) % n for n in (7, 3, 5, 11)], dim=1)
    rao_prior = torch.rand((counts.numel(), classes), generator=generator, dtype=dtype) + .1
    rao_prior = classes * rao_prior / rao_prior.sum(-1, keepdim=True)
    mode_prior = torch.ones((membership.numel(), modes), dtype=dtype)
    rao_alpha = rao_prior + torch.rand(rao_prior.shape, generator=generator, dtype=dtype) * 4
    mode_alpha = mode_prior + torch.rand(mode_prior.shape, generator=generator, dtype=dtype) * 3
    output = {"evidential": {
        "track_rao": {"alpha": rao_alpha.to(device), "prior_alpha": rao_prior.to(device)},
        "radar_mode": {"alpha": mode_alpha.to(device), "prior_alpha": mode_prior.to(device)},
    }}
    kwargs = dict(
        rao_labels=torch.randint(classes, (counts.numel(),), generator=generator).to(device),
        mode_labels=torch.randint(modes, (membership.numel(),), generator=generator).to(device),
        observation_to_track=membership.to(device), observation_counts=counts.to(device),
        component_ids=component_ids.to(device),
        component_weights=torch.tensor([.1, .2, .3, .4], dtype=dtype, device=device),
        rao_weights=torch.tensor([10 / 11, 1 / 11], dtype=dtype, device=device),
        rao_weight=.7, mode_weight=1.3,
    )
    return output, kwargs


def reference_loss(output, kwargs, coefficient):
    """Use direct per-track/component marginalization, without the shared helper."""
    ra = output["evidential"]["track_rao"]["alpha"]
    ma = output["evidential"]["radar_mode"]["alpha"]
    rp = output["evidential"]["track_rao"]["prior_alpha"]
    mp = output["evidential"]["radar_mode"]["prior_alpha"]
    rl, ml, components = kwargs["rao_labels"], kwargs["mode_labels"], kwargs["component_ids"]
    exact = torch.stack([torch.digamma(row.sum()) - torch.digamma(row[label])
                         for row, label in zip(ra, rl)])
    component = torch.stack([
        torch.stack([torch.digamma(row.sum()) - torch.digamma(
            row[components[:, index] == components[label, index]].sum())
                     for index in range(4)])
        for row, label in zip(ra, rl)
    ])
    rw = kwargs["rao_weights"]
    rao_edl = (rw[0] * exact + rw[1] * (component * kwargs["component_weights"]).sum(1)).mean()

    def balanced(values):
        membership = kwargs["observation_to_track"]
        return torch.stack([values[membership == index].mean()
                            for index in range(kwargs["observation_counts"].numel())]).mean()

    mode_edl = balanced(torch.digamma(ma.sum(-1)) - torch.digamma(
        ma.gather(1, ml[:, None]).squeeze(1)))

    def kl(alpha, prior, labels):
        mask = F.one_hot(labels, alpha.size(-1)).to(alpha.dtype)
        alpha = (alpha * (1 - mask) + prior * mask).float()
        prior = prior.float()
        total = alpha.sum(-1)
        return (torch.lgamma(total) - torch.lgamma(alpha).sum(-1)
                - torch.lgamma(prior.sum(-1)) + torch.lgamma(prior).sum(-1)
                + ((alpha - prior) * (torch.digamma(alpha)
                   - torch.digamma(total)[:, None])).sum(-1))

    rao_kl, mode_kl = kl(ra, rp, rl).mean(), balanced(kl(ma, mp, ml))
    parts = dict(rao_edl=rao_edl, mode_edl=mode_edl, rao_kl=rao_kl, mode_kl=mode_kl)
    for index, field in enumerate(("aircraft", "aircraft_type", "radar", "operator")):
        parts[f"rao_{field}_loss"] = kwargs["rao_weight"] * component[:, index].mean()
    loss = (kwargs["rao_weight"] * rao_edl + kwargs["mode_weight"] * mode_edl
            + coefficient * (rao_kl + mode_kl))
    return loss, parts


def differentiable(output):
    return {"evidential": {
        task: {key: value.detach().clone().requires_grad_(key == "alpha")
               for key, value in result.items()}
        for task, result in output["evidential"].items()
    }}


@pytest.mark.parametrize("counts,classes", [([1], 1), ([1, 7, 2], 17), ([3, 1, 9, 4, 2, 6, 1, 5], 122)])
@pytest.mark.parametrize("coefficient", [0.0, .00125, .005])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_loss_diagnostics_and_gradients_match_grouped_reference(counts, classes, coefficient, dtype):
    output, kwargs = make_case(counts, classes, dtype)
    baseline, candidate = differentiable(output), differentiable(output)
    expected_loss, expected_parts = reference_loss(baseline, kwargs, coefficient)
    actual_loss, actual_parts = batch_hierarchical_evidential_loss(
        candidate, **kwargs, kl_coefficient=coefficient,
    )
    torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-6, atol=2e-7)
    assert actual_parts.keys() == expected_parts.keys()
    for name, expected in expected_parts.items():
        torch.testing.assert_close(actual_parts[name], expected, rtol=2e-6, atol=2e-7)
    expected_loss.backward()
    actual_loss.backward()
    for task in ("track_rao", "radar_mode"):
        torch.testing.assert_close(candidate["evidential"][task]["alpha"].grad,
                                   baseline["evidential"][task]["alpha"].grad,
                                   rtol=3e-6, atol=2e-7)


def test_component_work_is_shared_and_training_can_skip_diagnostics():
    output, kwargs = make_case([1, 7, 2], 17)
    import rgcn_fusion.track_star_loss as loss_module
    with patch.object(loss_module, "component_expected_dirichlet_ce",
                      wraps=component_expected_dirichlet_ce) as components:
        diagnostic_loss, parts = batch_hierarchical_evidential_loss(output, **kwargs)
    assert components.call_count == 1
    assert len(parts) == 8
    plain_loss, plain_parts = batch_hierarchical_evidential_loss(
        output, **kwargs, return_diagnostics=False,
    )
    torch.testing.assert_close(plain_loss, diagnostic_loss, rtol=0, atol=0)
    assert plain_parts == {}


def test_precomputed_counts_do_not_recompute_bincount():
    output, kwargs = make_case([1, 7, 2], 17)
    with patch.object(torch, "bincount", side_effect=AssertionError("batch metadata is already prepared")):
        loss, parts = batch_hierarchical_evidential_loss(output, **kwargs)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())


def test_loss_supports_dynamic_compilation():
    compiled = torch.compile(batch_hierarchical_evidential_loss, dynamic=True, backend="eager")
    for counts in ([1, 4], [3, 2, 7]):
        output, kwargs = make_case(counts, 17)
        expected, _ = batch_hierarchical_evidential_loss(output, **kwargs, kl_coefficient=.005)
        actual, _ = compiled(output, **kwargs, kl_coefficient=.005)
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_loss_matches_cpu_without_host_reads():
    cpu_output, cpu_kwargs = make_case([1, 7, 2], 17)
    output, kwargs = make_case([1, 7, 2], 17, device="cuda")
    expected, _ = batch_hierarchical_evidential_loss(cpu_output, **cpu_kwargs, kl_coefficient=.005)
    # The helper must not fetch labels, counts, loss values or diagnostics to CPU.
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected host transfer")), \
         patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected scalar read")):
        actual, _ = batch_hierarchical_evidential_loss(output, **kwargs, kl_coefficient=.005)
    torch.testing.assert_close(actual.cpu(), expected)
