import ast
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "observation_series_and_intel_rgcn_classification_advanced_network_llm.ipynb"
)


def _explanation_functions():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"build_explanation_evidence", "explanation_prompt"}
    ]
    namespace = {
        "Any": Any,
        "np": np,
        "json": json,
        "TARGETS": ("radar_type",),
        "CRITICAL_EMITTER_TASKS": ("radar_type",),
        "EXTREME_UNCERTAINTY": 0.60,
        "HIGH_UNCERTAINTY": 0.35,
        "node_meta": [{"observation_id": "obs-1"}],
        "observation_node_indices": [0],
        "predictions": [{
            "observation_id": "obs-1",
            "kg_structured_prediction": {
                "status": "known",
                "labels": {"radar_type": "radar-a"},
                "confidence": {},
            },
        }],
        "label_vocab": {"radar_type": ["radar-a", "radar-b"]},
        "evidential_outputs": {
            "radar_type": {
                "evidence": np.array([[3.0, 1.0]]),
                "belief": np.array([[0.5, 1.0 / 6.0]]),
                "probabilities": np.array([[2.0 / 3.0, 1.0 / 3.0]]),
                "uncertainty": np.array([[1.0 / 3.0]]),
                "strength": np.array([[6.0]]),
            }
        },
    }
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_llm_packet_reports_modified_ds_frame_and_all_evidential_quantities():
    namespace = _explanation_functions()
    packet = namespace["build_explanation_evidence"](0)
    task = packet["tasks"]["radar_type"]
    leading = task["leading_hypotheses"][0]

    assert task["frame_of_discernment"]["hypotheses"] == ["radar-a", "radar-b"]
    assert task["frame_of_discernment"]["full_frame_uncertainty"] == 1.0 / 3.0
    assert leading["evidence"] == 3.0
    assert leading["belief"] == 0.5
    assert np.isclose(leading["plausibility"], 5.0 / 6.0)
    assert np.isclose(leading["uncertainty"], 1.0 / 3.0)
    assert np.isclose(leading["pignistic_probability"], 2.0 / 3.0)
    assert packet["kg_structured_prediction"]["labels"]["radar_type"] == "radar-a"


def test_llm_prompt_explains_frame_semantics_and_rejects_joint_ds_claims():
    namespace = _explanation_functions()
    prompt = namespace["explanation_prompt"](namespace["build_explanation_evidence"](0))

    assert "evidence is the GNN's non-negative support" in prompt
    assert "plausibility is belief plus the full-frame Theta mass" in prompt
    assert "pignistic probability shares the Theta mass equally" in prompt
    assert "not one learned joint frame" in prompt


def test_ollama_request_disables_thinking_to_require_a_response():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    tree = ast.parse(source)
    ollama_generate = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "ollama_generate"
    )
    payload = next(
        node.value
        for node in ast.walk(ollama_generate)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "payload" for target in node.targets)
        and isinstance(node.value, ast.Call)
    )
    request_options = next(
        node for node in ast.walk(payload) if isinstance(node, ast.Dict)
    )

    options = {
        key.value: value
        for key, value in zip(request_options.keys, request_options.values)
        if isinstance(key, ast.Constant)
    }
    assert isinstance(options["think"], ast.Constant)
    assert options["think"].value is False


def test_llm_explanation_ingests_one_positionally_selected_result():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert 'LLM_RESULT_INDEX = int(os.environ.get("LLM_RESULT_INDEX", "0"))' in source
    assert "selected_result = predictions[LLM_RESULT_INDEX]" in source
    assert "explanation = explain_emitter(LLM_RESULT_INDEX)" in source
    assert '"num_predict": OLLAMA_NUM_PREDICT' in source
    assert "LLM_EXPLANATION_WORKERS" not in source
