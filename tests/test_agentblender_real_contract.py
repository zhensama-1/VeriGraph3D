from pathlib import Path
import sys

from verigraph3d.agentblender import AgentBlenderActionMapper, AgentBlenderStateMapper
from verigraph3d.models import ActionType, SemanticAction


ABWS_SRC = Path(__file__).parents[1] / "external" / "AgentBlender-World-State" / "src"
sys.path.insert(0, str(ABWS_SRC))

from agent_blender.models import (  # noqa: E402
    Bounds, CameraState, Environment, LightState, ObjectState, TransactionSpec, WorldState,
)
from agent_blender.runtime import SceneRuntime  # noqa: E402


def real_world_state():
    cup = ObjectState(
        id="cup_01", blender_name="Cup", category="cup", display_name="Cup",
        dimensions=(0.4, 0.4, 0.5), transform={"location": (1, 0, 0)},
    )
    table = ObjectState(
        id="table_01", blender_name="Table", category="table", display_name="Table",
        dimensions=(3, 2, 1), transform={"location": (0, 2, 0)},
    )
    return WorldState(
        scene_id="contract",
        objects={cup.id: cup, table.id: table},
        environment=Environment(room_bounds=Bounds(min=(-5, -5, 0), max=(5, 5, 5))),
    )


def test_mapper_accepts_real_abws_pydantic_world_state():
    mapped = AgentBlenderStateMapper().map(real_world_state())
    assert set(mapped.objects) == {"Cup", "Table"}
    assert mapped.objects["Cup"].metadata["abws_id"] == "cup_01"
    assert mapped.objects["Cup"].aabb.minimum[2] == 0
    assert mapped.objects["Cup"].location[2] == 0.25


def test_action_mapper_builds_real_transaction_spec_and_runtime_accepts_it():
    world = real_world_state()
    mapped = AgentBlenderStateMapper().map(world)
    payload = AgentBlenderActionMapper().map(
        SemanticAction(
            "move_contract", ActionType.MOVE_OBJECT, "Cup",
            parameters={"location": [2, 0, 0.25]},
        ),
        mapped,
    )
    specification = TransactionSpec.parse_obj(payload)
    result = SceneRuntime(world).execute_transaction(specification)
    assert result.success
    assert result.diff.modified == ["cup_01"]
    assert result.state.objects["cup_01"].transform.location == (2.0, 0.0, 0.0)


def test_real_camera_light_and_color_contracts():
    world = real_world_state()
    world.cameras["camera_main"] = CameraState(
        id="camera_main", blender_name="Camera", transform={"location": (5, -5, 4)}
    )
    world.lights["key_light"] = LightState(
        id="key_light", blender_name="Key", type="AREA", energy=800
    )
    mapped = AgentBlenderStateMapper().map(world)
    cases = [
        SemanticAction(
            "camera_contract", ActionType.SET_CAMERA, "Camera",
            parameters={"fov_degrees": 60},
        ),
        SemanticAction(
            "light_contract", ActionType.SET_LIGHT, "Key",
            parameters={"energy": 1200},
        ),
        SemanticAction(
            "color_contract", ActionType.SET_MATERIAL, "Cup",
            parameters={"material": "cup_red", "color": [1, 0, 0, 1]},
        ),
    ]
    runtime = SceneRuntime(world)
    for action in cases:
        specification = TransactionSpec.parse_obj(AgentBlenderActionMapper().map(action, mapped))
        result = runtime.execute_transaction(specification)
        assert result.success, [(issue.code, issue.message) for issue in result.validation]
        mapped = AgentBlenderStateMapper().map(runtime.state)
    assert runtime.state.cameras["camera_main"].fov_degrees == 60
    assert runtime.state.lights["key_light"].energy == 1200
    assert runtime.state.objects["cup_01"].materials[0].material_id == "cup_red"


def test_real_create_then_delete_contract():
    runtime = SceneRuntime(real_world_state())
    mapped = AgentBlenderStateMapper().map(runtime.state)
    create = SemanticAction(
        "create_contract", ActionType.CREATE_OBJECT, "vase_01",
        parameters={
            "primitive": "cylinder", "category": "vase",
            "dimensions": [0.4, 0.4, 0.8], "location": [0.0, 0.0, 0.4],
            "color": [0.1, 0.3, 0.8, 1.0],
        },
    )
    create_spec = TransactionSpec.parse_obj(
        AgentBlenderActionMapper().map(create, mapped)
    )
    created = runtime.execute_transaction(create_spec)
    assert created.success
    assert created.diff.created == ["vase_01"]
    assert runtime.state.objects["vase_01"].transform.location == (0.0, 0.0, 0.0)
    assert "delete" in runtime.state.objects["vase_01"].capabilities

    mapped = AgentBlenderStateMapper().map(runtime.state)
    delete = SemanticAction(
        "delete_contract", ActionType.DELETE_OBJECT, "ABWS__vase_01"
    )
    delete_spec = TransactionSpec.parse_obj(
        AgentBlenderActionMapper().map(delete, mapped)
    )
    deleted = runtime.execute_transaction(delete_spec)
    assert deleted.success
    assert deleted.diff.deleted == ["vase_01"]
