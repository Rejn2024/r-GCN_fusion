import numpy as np

from rgcn_fusion.dempster_shafer import belief_plausibility, combine_masses


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
