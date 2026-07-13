import numpy as np
from rgcn_fusion.dempster_shafer import belief_plausibility, combine_masses, subset_masks


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


def test_subset_masks_use_singletons_plus_uncertainty_for_more_than_ten_hypotheses():
    assert subset_masks(11) == [1 << idx for idx in range(11)] + [(1 << 11) - 1]


def test_belief_plausibility_accepts_singleton_uncertainty_masses_for_more_than_ten_hypotheses():
    hypotheses = [f"h{i}" for i in range(11)]
    intervals = belief_plausibility([0.75] + [0.0] * 10 + [0.25], hypotheses)
    assert intervals[0].belief == 0.75
    assert intervals[0].plausibility == 1.0
    assert all(interval.belief == 0.0 for interval in intervals[1:])
    assert all(interval.plausibility == 0.25 for interval in intervals[1:])


def test_combine_masses_accepts_singleton_uncertainty_vectors():
    combined = combine_masses([0.7, 0.2] + [0.0] * 9 + [0.1], [0.6, 0.3] + [0.0] * 9 + [0.1])
    assert np.isclose(combined.sum(), 1.0)
    assert combined.shape == (12,)
    assert combined[0] > combined[1]
    assert combined[-1] > 0.0
