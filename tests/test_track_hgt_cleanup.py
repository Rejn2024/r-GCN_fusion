"""Exercise post-training consumers after the notebook's actual cleanup cell."""
import ast
import gc
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "track_hgt.ipynb"


class TrackHGTCleanupTest(unittest.TestCase):
    def test_cleanup_preserves_partial_identification_and_export_inputs(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        sources = ["".join(c["source"]) for c in notebook["cells"]
                   if c["cell_type"] == "code"]
        cleanup = next(s for s in sources if "_CONSTRUCTION_ONLY_OBJECTS = (" in s)
        policy = next(s for s in sources if s.startswith(
            "# Major step: Maximal safe partial-identification policy"))
        export = next(s for s in sources if s.startswith(
            "# Major step: Fast final evaluation and export\n"))
        worlds = [{"aircraft_variant": "A", "aircraft_family": "Family A",
                   "radar_type": "R", "operator_country": "O"}]
        observation_tracks = torch.tensor([1, 0, 1], dtype=torch.long)
        # CUDA cleanup is unrelated to this regression; keep this check CPU-only.
        namespace = dict(gc=gc, torch=SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: False)), kg_valid_worlds=worlds,
            observation_track_index_cpu=observation_tracks, series_records=[{}])
        exec(compile(cleanup, str(NOTEBOOK), "exec"), namespace)
        self.assertNotIn("series_records", namespace)
        self.assertIs(namespace["kg_valid_worlds"], worlds)
        self.assertIs(namespace["observation_track_index_cpu"], observation_tracks)

        with TemporaryDirectory() as directory:
            kg_path = Path(directory) / "kg.json"
            kg_path.write_text('{"nodes": [], "edges": []}', encoding="utf-8")
            namespace.update(torch=torch, np=np, os=os, hashlib=hashlib,
                             defaultdict=defaultdict, KG_JSON=kg_path,
                             rao_vocab=[("A", "R", "O")])
            # Run the full downstream policy cell, including its taxonomy loop.
            exec(compile(policy, str(NOTEBOOK), "exec"), namespace)
            self.assertEqual(namespace["rao_world_records"][0]["aircraft_type"], "Family A")
            partial = namespace["maximal_safe_partial_identification"]([1.0], [0.9], 0.1)
            self.assertEqual(partial["status"], "known")

            namespace.update(DEVICE=torch.device("cpu"),
                             partial_identifications=[partial, partial],
                             rao_mode_compatibility=torch.tensor([[1., 0.]]),
                             observation_mode_mask=torch.zeros((3, 2), dtype=torch.bool),
                             positions_by_track=[[], []])
            # Execute both real export loops that consume the CPU track mapping.
            loops = [node for node in ast.parse(export).body
                     if isinstance(node, ast.For)
                     and "observation_track_index_cpu" in ast.unparse(node.iter)]
            self.assertEqual(len(loops), 2)
            exec(compile(ast.Module(body=loops, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
            self.assertEqual(namespace["positions_by_track"], [[1], [0, 2]])
            self.assertEqual(namespace["observation_mode_mask"].tolist(),
                             [[True, False], [True, False], [True, False]])


if __name__ == "__main__":
    unittest.main()
