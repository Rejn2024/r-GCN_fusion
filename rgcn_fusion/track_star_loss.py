"""Track-balanced evidential losses for the HGT star experiment.

The batch's labels, memberships and observation counts are prepared once on the
active device.  Joint-RAO component expectations are shared by the objective and
optional diagnostics rather than recomputed for each consumer.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


COMPONENT_DIAGNOSTICS = (
    "rao_aircraft_loss", "rao_aircraft_type_loss", "rao_radar_loss", "rao_operator_loss",
)


def expected_dirichlet_ce(alpha: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Expected categorical NLL for each labelled Dirichlet opinion."""
    return (torch.digamma(alpha.sum(-1))
            - torch.digamma(alpha.gather(1, labels[:, None]).squeeze(1)))


def component_expected_dirichlet_ce(
    alpha: torch.Tensor, labels: torch.Tensor, component_ids: torch.Tensor,
) -> torch.Tensor:
    """Expected NLLs of each true component, marginalized over valid RAO worlds."""
    true_components = component_ids[labels]
    total_digamma = torch.digamma(alpha.sum(dim=1))
    component_losses = []
    for component_index in range(component_ids.size(1)):
        matching_worlds = (component_ids[:, component_index].unsqueeze(0)
                           == true_components[:, component_index].unsqueeze(1))
        matching_evidence = (alpha * matching_worlds.to(alpha.dtype)).sum(dim=1)
        component_losses.append(total_digamma - torch.digamma(matching_evidence))
    return torch.stack(component_losses, dim=1)


def dirichlet_kl_to_prior(alpha: torch.Tensor, prior_alpha: torch.Tensor) -> torch.Tensor:
    """KL from an opinion to its prior, retaining FP32 evidential arithmetic."""
    alpha, prior_alpha = alpha.float(), prior_alpha.float()
    total, prior_total = alpha.sum(-1), prior_alpha.sum(-1)
    return (torch.lgamma(total) - torch.lgamma(alpha).sum(-1)
            - torch.lgamma(prior_total) + torch.lgamma(prior_alpha).sum(-1)
            + ((alpha - prior_alpha) * (torch.digamma(alpha)
               - torch.digamma(total).unsqueeze(-1))).sum(-1))


def suppress_true_class_evidence(
    alpha: torch.Tensor, labels: torch.Tensor, prior_alpha: torch.Tensor,
) -> torch.Tensor:
    """Restore the true class to its prior before evidence regularization."""
    mask = F.one_hot(labels, alpha.size(-1)).to(alpha.dtype)
    return alpha * (1.0 - mask) + prior_alpha * mask


def track_balanced_mean(
    values: torch.Tensor, observation_to_track: torch.Tensor, observation_counts: torch.Tensor,
) -> torch.Tensor:
    """Average within each track, then equally across tracks.

    Counts are immutable batch metadata, shared by mode EDL and KL.  In
    particular, this is not an observation-weighted mean on unequal tracks.
    """
    totals = values.new_zeros(observation_counts.numel())
    totals.index_add_(0, observation_to_track, values.to(totals.dtype))
    return (totals / observation_counts.to(values.dtype).clamp_min(1)).mean()


def batch_hierarchical_evidential_loss(
    outputs: Mapping,
    *,
    rao_labels: torch.Tensor,
    mode_labels: torch.Tensor,
    observation_to_track: torch.Tensor,
    observation_counts: torch.Tensor,
    component_ids: torch.Tensor,
    component_weights: torch.Tensor,
    rao_weights: torch.Tensor,
    kl_coefficient: float = 0.0,
    rao_weight: float = 1.0,
    mode_weight: float = 1.0,
    return_diagnostics: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the joint-RAO plus track-balanced mode objective and diagnostics.

    ``rao_weights`` contains normalized complete/component contributions and
    ``component_weights`` contains normalized variant/type/radar/operator
    contributions, matching the notebook configuration.  Inputs already reside
    on the active device; this function performs no host transfers or scalar
    reads.  Training can skip diagnostic means while preserving the objective.
    """
    rao = outputs["evidential"]["track_rao"]
    mode = outputs["evidential"]["radar_mode"]
    rao_alpha, mode_alpha = rao["alpha"], mode["alpha"]
    component_matrix = component_expected_dirichlet_ce(rao_alpha, rao_labels, component_ids)
    complete_loss = expected_dirichlet_ce(rao_alpha, rao_labels)
    component_loss = (component_matrix * component_weights).sum(dim=1)
    rao_edl = (rao_weights[0] * complete_loss + rao_weights[1] * component_loss).mean()
    mode_edl = track_balanced_mean(
        expected_dirichlet_ce(mode_alpha, mode_labels), observation_to_track, observation_counts,
    )
    rao_kl = dirichlet_kl_to_prior(
        suppress_true_class_evidence(rao_alpha, rao_labels, rao["prior_alpha"]), rao["prior_alpha"],
    ).mean()
    mode_kl = track_balanced_mean(
        dirichlet_kl_to_prior(
            suppress_true_class_evidence(mode_alpha, mode_labels, mode["prior_alpha"]),
            mode["prior_alpha"],
        ), observation_to_track, observation_counts,
    )
    loss = rao_weight * rao_edl + mode_weight * mode_edl + kl_coefficient * (rao_kl + mode_kl)
    if not return_diagnostics:
        return loss, {}
    component_means = component_matrix.mean(dim=0)
    components = {name: rao_weight * component_means[index]
                  for index, name in enumerate(COMPONENT_DIAGNOSTICS)}
    return loss, {"rao_edl": rao_edl, "mode_edl": mode_edl,
                  "rao_kl": rao_kl, "mode_kl": mode_kl, **components}
