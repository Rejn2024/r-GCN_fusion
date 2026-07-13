import numpy as np
from rgcn_fusion.dempster_shafer import belief_plausibility, combine_masses, grouped_type_masks, subset_masks


def test_belief_plausibility_for_two_hypotheses():
    intervals = belief_plausibility([0.2, 0.3, 0.5], ["a", "b"])
    assert intervals[0].belief == 0.2
    assert intervals[0].plausibility == 0.7
    assert intervals[1].belief == 0.3
    assert intervals[1].plausibility == 0.8


def test_combine_masses_normalizes_conflict():
    combined = combine_masses([0.6, 0.1, 0.3], [0.2, 0.7, 0.1])
    assert np.isclose(combined.sum(), 1.0)
    assert combined.shape == (3,)


def test_subset_masks_use_singletons_type_groups_plus_uncertainty_for_more_than_ten_hypotheses():
    hypotheses = [
        "aircraft:mig_29a",
        "aircraft:mig_29s",
        "aircraft:mig_29smt",
        "aircraft:mig_29k",
        "aircraft:mig_35",
        "aircraft:su_30mki",
        "aircraft:su_30sm",
        "aircraft:f_16a_b",
        "aircraft:f_16c_d_block_50",
        "aircraft:f_15c",
        "aircraft:rafale_c",
    ]
    assert grouped_type_masks(hypotheses) == [0b1111, (1 << 5) | (1 << 6), (1 << 7) | (1 << 8)]
    assert subset_masks(hypotheses) == [1 << idx for idx in range(11)] + grouped_type_masks(hypotheses) + [(1 << 11) - 1]


def test_belief_plausibility_accepts_singleton_type_group_uncertainty_masses_for_more_than_ten_hypotheses():
    hypotheses = [f"aircraft:mig_29{suffix}" for suffix in ["a", "s", "smt", "k"]] + [f"aircraft:h{i}" for i in range(7)]
    intervals = belief_plausibility([0.75] + [0.0] * 10 + [0.0] + [0.25], hypotheses)
    assert intervals[0].belief == 0.75
    assert intervals[0].plausibility == 1.0
    assert all(interval.belief == 0.0 for interval in intervals[1:])
    assert all(interval.plausibility == 0.25 for interval in intervals[1:])


def test_combine_masses_accepts_compact_singleton_uncertainty_vectors():
    combined = combine_masses([0.7, 0.2] + [0.0] * 9 + [0.1], [0.6, 0.3] + [0.0] * 9 + [0.1])
    assert np.isclose(combined.sum(), 1.0)
    assert combined.shape == (12,)
    assert combined[0] > combined[1]
    assert combined[-1] > 0.0


def test_combine_masses_accepts_type_group_vectors_with_hypotheses():
    hypotheses = [f"aircraft:mig_29{suffix}" for suffix in ["a", "s", "smt", "k"]] + [f"aircraft:h{i}" for i in range(7)]
    left = [0.7] + [0.0] * 10 + [0.2] + [0.1]
    right = [0.6] + [0.0] * 10 + [0.3] + [0.1]
    combined = combine_masses(left, right, hypotheses)
    assert np.isclose(combined.sum(), 1.0)
    assert combined.shape == (13,)
    assert combined[0] > combined[-2]
