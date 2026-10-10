"""Exercise notebook gathering without building graphs or training models."""

import ast
import json
from pathlib import Path

import pytest
import torch


NOTEBOOKS = sorted(Path("notebooks").glob("Track*.ipynb"))


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
@pytest.mark.parametrize("metadata_count", [0, 1, 5])
@pytest.mark.parametrize("selected_only", [False, True])
def test_gathering_uses_persistent_observation_order(notebook, metadata_count, selected_only):
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(notebook.read_text())["cells"]
        if cell["cell_type"] == "code"
    )
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "model_forward"
    )
    # Noncontiguous, out-of-order positions ensure scatter preserves global order.
    batches = [
        ({}, None, (None, None, torch.tensor([1]), torch.tensor([3, 1]))),
        ({}, None, (None, None, torch.tensor([0]), torch.tensor([4, 0, 2]))),
    ]

    def forward(batch):
        tracks, positions = batch[2][2:]
        track_values = tracks.float()[:, None].repeat(1, 2)
        observation_values = positions.float()[:, None].repeat(1, 2)
        return {
            "track_embeddings": track_values,
            "observation_embeddings": observation_values,
            "track_rao": track_values,
            "radar_mode": observation_values,
            "evidential": {
                "track_rao": {"probabilities": track_values, "uncertainty": track_values[:, :1]},
                "radar_mode": {"probabilities": observation_values, "uncertainty": observation_values[:, :1]},
            },
        }

    namespace = {
        "DEVICE": torch.device("cpu"),
        "series_ids": ["track-0", "track-1"],
        "rao_vocab": ["a", "b"],
        "mode_vocab": ["a", "b"],
        "observation_node_indices": list(range(metadata_count)),
        "observation_track_index": torch.tensor([0, 1, 0, 1, 0]),
        "prepared_batches_by_split": {"train": batches[:1], "test": batches[1:], "val": []},
        "model_forward_batch": forward,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(notebook), "exec"), namespace)
    output = namespace["model_forward"](batches[:1]) if selected_only else namespace["model_forward"]()
    positions = torch.tensor([3, 1]) if selected_only else torch.arange(5)
    tracks = torch.tensor([1]) if selected_only else torch.arange(2)
    assert output["observation_embeddings"].shape == (5, 2)
    torch.testing.assert_close(output["observation_embeddings"][positions], positions.float()[:, None].repeat(1, 2))
    torch.testing.assert_close(output["track_embeddings"][tracks], tracks.float()[:, None].repeat(1, 2))
    for key in ("probabilities", "uncertainty"):
        values = output["evidential"]["radar_mode"][key]
        assert values.size(0) == 5
        torch.testing.assert_close(values[positions], positions.float()[:, None].expand(-1, values.size(1)))
    if "self.rao_head" in source:
        assert output["radar_mode"].shape == (5, 2)
        torch.testing.assert_close(output["radar_mode"][positions], positions.float()[:, None].repeat(1, 2))
