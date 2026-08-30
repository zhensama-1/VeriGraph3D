"""Build an occlusion scene and verify active view selection with Blender ray casts."""

from enum import Enum
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from verigraph3d.active_view import ActiveViewSelector  # noqa: E402
from verigraph3d.backends import BlenderBackend  # noqa: E402
from verigraph3d.execution import SafeExecutor  # noqa: E402
from verigraph3d.models import ActionType, SemanticAction  # noqa: E402


def main() -> None:
    output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.4, depth=1.5, location=(0, 0, 1.0))
    bpy.context.object.name = "Cup"
    bpy.ops.mesh.primitive_cube_add(location=(0, -3, 1.2))
    blocker = bpy.context.object
    blocker.name = "Blocker"
    blocker.scale = (2.5, 0.5, 2.5)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.camera_add(location=(0, -8, 1.2))
    camera = bpy.context.object
    camera.name = "Camera"
    camera.rotation_euler = (Vector((0, 0, 1.0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    bpy.context.view_layer.update()

    backend = BlenderBackend()
    before = backend.read_state()
    before_score = before.objects["Cup"].metadata["camera_visibility"]["Camera"]
    uncertainty = [{"subject": "Cup", "predicate": "visible_from", "confidence": before_score}]
    candidate, gain = ActiveViewSelector().select(before, uncertainty, 0.1, backend.score_view)
    if candidate is None:
        raise RuntimeError("No useful active view found")
    action = SemanticAction(
        "observation_001", ActionType.SET_CAMERA, "Camera",
        parameters={"location": candidate.location, "target": candidate.target, "fov_degrees": candidate.fov_degrees},
    )
    record = SafeExecutor(backend).execute(action)
    after = backend.read_state()
    after_score = after.objects["Cup"].metadata["camera_visibility"]["Camera"]
    payload = {
        "before_visibility": before_score,
        "after_visibility": after_score,
        "information_gain": gain,
        "camera_location": candidate.location,
        "action_status": record.status,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("VERIGRAPH_VISIBILITY", json.dumps(payload))


if __name__ == "__main__":
    main()
