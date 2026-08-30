import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY = Path(__file__).parents[1]


@pytest.mark.blender
def test_real_blender_end_to_end(tmp_path):
    blender = os.environ.get("BLENDER_EXECUTABLE")
    if not blender:
        pytest.skip("Set BLENDER_EXECUTABLE to run the real Blender integration test")
    fixture = tmp_path / "fixture.blend"
    create = subprocess.run(
        [blender, "--background", "--factory-startup", "--python", str(REPOSITORY / "scripts" / "create_blender_fixture.py"), "--", str(fixture)],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    assert create.returncode == 0, create.stdout + create.stderr
    output, checkpoint = tmp_path / "run.json", tmp_path / "before.blend"
    run = subprocess.run(
        [
            blender, "--background", str(fixture), "--python", str(REPOSITORY / "scripts" / "run_blender_task.py"), "--",
            str(REPOSITORY / "examples" / "tasks.json"), "place_and_color_001",
            "--output", str(output), "--checkpoint", str(checkpoint),
        ],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["accepted"]
    assert result["final_state"]["objects"]["Cup"]["color"] == [1.0, 0.0, 0.0, 1.0]
    assert result["verification"]["deterministic"][0]["passed"]
    assert checkpoint.is_file()

    reference = tmp_path / "reference.png"
    render_reference = subprocess.run(
        [blender, "--background", str(fixture), "--python", str(REPOSITORY / "scripts" / "create_reference_render.py"), "--", str(reference)],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    assert render_reference.returncode == 0, render_reference.stdout + render_reference.stderr
    dataset_payload = json.loads((REPOSITORY / "examples" / "tasks.json").read_text(encoding="utf-8"))
    dataset_payload["tasks"][0]["reference_images"] = [str(reference)]
    visual_dataset = tmp_path / "tasks_with_reference.json"
    visual_dataset.write_text(json.dumps(dataset_payload, ensure_ascii=False), encoding="utf-8")
    visual_output = tmp_path / "visual_run.json"
    visual_run = subprocess.run(
        [
            blender, "--background", str(fixture), "--python", str(REPOSITORY / "scripts" / "run_blender_task.py"), "--",
            str(visual_dataset), "place_and_color_001", "--output", str(visual_output),
            "--render-directory", str(tmp_path / "renders"),
        ],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    assert visual_run.returncode == 0, visual_run.stdout + visual_run.stderr
    visual_result = json.loads(visual_output.read_text(encoding="utf-8"))
    assert visual_result["accepted"]
    assert visual_result["metrics"]["renders"] == 2
    assert visual_result["verification"]["visual_score"] > 0.99
    assert visual_result["trace"][0]["verification"]["visual_score"] < 0.8
    assert visual_result["trace"][1]["verification"]["visual_score"] > 0.99

    visibility_output = tmp_path / "visibility.json"
    visibility = subprocess.run(
        [
            blender, "--background", "--factory-startup", "--python",
            str(REPOSITORY / "scripts" / "probe_blender_visibility.py"), "--", str(visibility_output),
        ],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    assert visibility.returncode == 0, visibility.stdout + visibility.stderr
    visibility_result = json.loads(visibility_output.read_text(encoding="utf-8"))
    assert visibility_result["action_status"] == "success"
    assert visibility_result["before_visibility"] < 0.1
    assert visibility_result["after_visibility"] > 0.5
    assert visibility_result["information_gain"] > 0


@pytest.mark.blender
def test_agentblender_atomic_candidate_preserves_source(tmp_path):
    blender = os.environ.get("BLENDER_EXECUTABLE")
    if not blender:
        pytest.skip("Set BLENDER_EXECUTABLE to run the real Blender integration test")
    abws_root = REPOSITORY / "external" / "AgentBlender-World-State"
    sys.path.insert(0, str(abws_root / "src"))
    from verigraph3d.agent import VeriGraph3DAgent
    from verigraph3d.agentblender import AgentBlenderAtomicBlendBackend
    from verigraph3d.models import ActionType, Constraint, GoalSpec, SemanticAction

    source = tmp_path / "abws_source.blend"
    created = subprocess.run(
        [
            blender, "--background", "--python",
            str(abws_root / "blender_scripts" / "create_test_scene.py"),
            "--", "--output", str(source),
        ],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    assert created.returncode == 0, created.stdout + created.stderr
    backend = AgentBlenderAtomicBlendBackend(
        source, tmp_path / "transactions", blender_executable=blender
    )
    result = VeriGraph3DAgent(backend).run(goal=GoalSpec(required=[
        Constraint(
            "ABWS__chair_01", "location",
            expected=[1.5, 0.0, 0.455], tolerance=0.001,
        )
    ]))
    destination = backend.export(tmp_path / "accepted.blend")
    before = backend.process.extract(source, tmp_path / "before.json", "before")
    after = backend.process.extract(destination, tmp_path / "after.json", "after")
    assert result["accepted"]
    assert before.revision == 0
    assert before.objects["chair_01"].transform.location[0] == 2.0
    assert after.revision == 1
    assert after.objects["chair_01"].transform.location[0] == 1.5

    backend.apply(SemanticAction(
        "create_vase", ActionType.CREATE_OBJECT, "vase_01",
        parameters={
            "primitive": "cylinder", "category": "vase",
            "dimensions": [0.4, 0.4, 0.8], "location": [0.0, -2.0, 0.4],
            "color": [0.1, 0.3, 0.8, 1.0],
        },
    ))
    created_state = backend.read_state()
    assert created_state.metadata["abws_revision"] == 2
    assert created_state.objects["ABWS__vase_01"].dimensions == pytest.approx((0.4, 0.4, 0.8))
    assert created_state.metadata["abws_last_transaction"]["status"] == "COMMITTED"

    backend.apply(SemanticAction(
        "delete_vase", ActionType.DELETE_OBJECT, "ABWS__vase_01"
    ))
    deleted_state = backend.read_state()
    unchanged_source = backend.process.extract(source, tmp_path / "source_final.json", "source")
    assert deleted_state.metadata["abws_revision"] == 3
    assert "ABWS__vase_01" not in deleted_state.objects
    assert unchanged_source.revision == 0
    assert "vase_01" not in unchanged_source.objects
