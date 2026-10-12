"""Dependency-free regression checks for notebook open-set partitioning."""
import ast
import json
from pathlib import Path
import unittest

NOTEBOOK = Path('notebooks/Track_identification_open_set_ufo_evidential_only_no_unknown_output_non_uniform_base_rate.ipynb')


class OpenSetPartitionTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads(NOTEBOOK.read_text())
        self.source = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        self.tree = ast.parse(self.source)

    def test_unknown_tracks_and_observations_cannot_enter_supervised_splits(self):
        function = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reserve_open_set_split')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
        known = {'train': {'a'}, 'test': {'b', 'ufo'}, 'val': {'c'}}
        splits = namespace['reserve_open_set_split'](known, {'ufo'})
        observations = [('a', 0), ('b', 1), ('ufo', 2), ('ufo', 3), ('c', 4)]
        positions = {name: [p for sid, p in observations if sid in tracks] for name, tracks in splits.items()}
        self.assertEqual(splits['test'], {'b'})
        self.assertEqual(positions['test'], [1])
        self.assertEqual(positions['open_set'], [2, 3])
        self.assertEqual(set.union(*splits.values()), {'a', 'b', 'c', 'ufo'})
        self.assertEqual(known['test'], {'b', 'ufo'})
        self.assertEqual(namespace['reserve_open_set_split']({'test': {'b'}}, set())['open_set'], set())

    def test_supervised_metrics_guard_and_full_inference_coverage(self):
        for name, argument in [('hierarchical_loss', 'split_name'), ('metrics_from_outputs', 'name')]:
            function = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            namespace = {'SUPERVISED_SPLITS': ('train', 'test', 'val')}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
            with self.assertRaisesRegex(ValueError, 'no supervised class labels'):
                namespace[name]({}, 'open_set')
        self.assertIn('for name in prepared_batches_by_split', self.source)
        self.assertIn('for name in SUPERVISED_SPLITS}', self.source)
        self.assertNotIn('split_series["test"].update(ufo_series_ids)', self.source)
        self.assertIn('rao_class_index.get(rao_target_by_series[series_id], -1)', self.source)
        self.assertIn('"open_set_evaluation": open_set_evaluation', self.source)


if __name__ == '__main__':
    unittest.main()
