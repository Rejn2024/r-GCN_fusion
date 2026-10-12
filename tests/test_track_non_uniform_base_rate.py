"""Verify order-of-battle priors and Dirichlet mapping without training."""
import ast
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

NOTEBOOK = Path('notebooks/Track_identification_open_set_ufo_evidential_only_no_unknown_output_non_uniform_base_rate.ipynb')


def functions():
    notebook = json.loads(NOTEBOOK.read_text())
    source = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    namespace = dict(np=np, torch=torch, math=math, F=F)
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == 'TrackRGCNHGTClassifier':
            node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'dirichlet_ds_output')
            node.decorator_list = []
        if isinstance(node, ast.FunctionDef) and node.name in {
            'order_of_battle_base_rates', 'dirichlet_ds_output',
            'dirichlet_kl_to_prior', 'suppress_true_class_evidence',
        }:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
    return namespace


def test_oob_membership_expands_valid_tuples_without_truth():
    f = functions()['order_of_battle_base_rates']
    vocabulary = [('A', 'R1', 'O1'), ('A', 'R2', 'O2'), ('B', 'R1', 'O1')]
    series = {'ground_truth_track_label': {'aircraft_variant': 'B'}, 'intelligence_reports': [
        {'report_type': 'theatre_aircraft_report', 'theatre_aircraft': {'aircraft_ids': ['aircraft:a']},
         'claims': [{'claim_type': 'theatre_aircraft_presence', 'object_ids': ['aircraft:a']}]},
        {'report_type': 'airborne_aircraft_report', 'theatre_aircraft': {'aircraft_types': ['B']}},
    ]}
    rates, audit = f(series, vocabulary, {'aircraft:a': 'A'})
    np.testing.assert_allclose(rates, [10 / 21, 10 / 21, 1 / 21])
    assert audit['boosted_class_indices'] == [0, 1]
    for control in [{}, {'intelligence_reports': [{'report_type': 'theatre_aircraft_report',
                                                  'theatre_aircraft': {'aircraft_types': ['X']}}]}]:
        uniform, audit = f(control, vocabulary, {})
        np.testing.assert_allclose(uniform, [1 / 3] * 3)
        assert audit['fallback_uniform']
    uniform, _ = f(series, vocabulary, {'aircraft:a': 'A'}, multiplier=1)
    np.testing.assert_allclose(uniform, [1 / 3] * 3)


def test_opinion_and_kl_match_dirichlet_distribution():
    ns = functions()
    rates = torch.tensor([[10 / 21, 10 / 21, 1 / 21], [1 / 3] * 3])
    out = ns['dirichlet_ds_output'](torch.tensor([[1., 2., 3.], [-100., -100., -100.]]), rates)
    torch.testing.assert_close(out['probabilities'], out['belief'] + out['uncertainty'] * rates)
    torch.testing.assert_close(out['belief'].sum(-1, keepdim=True) + out['uncertainty'], torch.ones(2, 1))
    torch.testing.assert_close(out['prior_alpha'].sum(-1), torch.full((2,), 3.))
    torch.testing.assert_close(out['probabilities'][1], rates[1])
    alpha, prior = out['alpha'], out['prior_alpha']
    expected = torch.distributions.kl_divergence(torch.distributions.Dirichlet(alpha), torch.distributions.Dirichlet(prior))
    torch.testing.assert_close(ns['dirichlet_kl_to_prior'](alpha, prior), expected)
    torch.testing.assert_close(ns['dirichlet_kl_to_prior'](prior, prior), torch.zeros(2))
    labels = torch.tensor([0, 2])
    suppressed = ns['suppress_true_class_evidence'](alpha, labels, prior)
    torch.testing.assert_close(suppressed[torch.arange(2), labels], prior[torch.arange(2), labels])
    mask = ~F.one_hot(labels, 3).bool()
    torch.testing.assert_close(suppressed[mask], alpha[mask])
