import json

from verigraph3d.agentblender import (
    AgentBlenderActionMapper,
    AgentBlenderJsonReader,
    AgentBlenderRuntimeBridge,
    AgentBlenderStateMapper,
    AgentBlenderTransactionBackend,
    AgentBlenderWorldStateBackend,
)
from verigraph3d.backends import MemoryBackend
from verigraph3d.execution import SafeExecutor
from verigraph3d.models import ActionType, Constraint, SemanticAction
from verigraph3d.recovery import FailureAnalyzer
from verigraph3d.models import VerificationReport


def abws_payload():
    return {
        "scene_id": "room-1",
        "revision": 7,
        "objects": [
            {
                "id": "table-uuid",
                "name": "Table",
                "category": "furniture",
                "transform": {"translation": [0, 0, 0.75], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
                "aabb": {"min": [-1.5, -1, 0], "max": [1.5, 1, 1.5]},
            },
            {
                "id": "cup-uuid",
                "name": "Cup",
                "category": "container",
                "parent_id": "table-uuid",
                "transform": {"position": {"x": 0, "y": 0, "z": 1.7}},
                "aabb": {"minimum": [-0.2, -0.2, 1.5], "maximum": [0.2, 0.2, 1.9]},
                "material": {"id": "red", "base_color": [1, 0, 0, 1]},
            },
        ],
        "cameras": {"camera-uuid": {"name": "Camera", "location": [6, -6, 5], "fov": 55}},
        "lights": [{"id": "light-uuid", "name": "Key", "type": "area", "power": 800}],
        "relations": [{"subject": "cup-uuid", "predicate": "on_top_of", "object": "table-uuid"}],
    }


def test_maps_abws_ids_revision_geometry_and_material():
    state = AgentBlenderStateMapper().map(abws_payload())
    assert state.timestamp == 7
    assert state.metadata["abws_scene_id"] == "room-1"
    assert state.objects["Cup"].metadata["abws_id"] == "cup-uuid"
    assert state.objects["Cup"].parent == "Table"
    assert state.objects["Cup"].aabb.minimum == (-0.2, -0.2, 1.5)
    assert state.objects["Cup"].material == "red"
    assert state.objects["Cup"].color == (1.0, 0.0, 0.0, 1.0)
    assert state.cameras["Camera"].fov_degrees == 55
    assert state.lights["Key"].light_type == "AREA"


def test_reads_abws_json_snapshot(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(abws_payload()), encoding="utf-8")
    state = AgentBlenderStateMapper().map(AgentBlenderJsonReader(path)())
    assert set(state.objects) == {"Table", "Cup"}


def test_hybrid_backend_delegates_writes_and_uses_abws_reads():
    mapped = AgentBlenderStateMapper().map(abws_payload())
    writer = MemoryBackend(mapped)
    backend = AgentBlenderWorldStateBackend(writer, lambda: writer.read_state().to_dict())
    backend.apply(SemanticAction("a1", ActionType.MOVE_OBJECT, "Cup", parameters={"location": [1, 2, 3]}))
    assert backend.read_state().objects["Cup"].location == (1.0, 2.0, 3.0)


def test_hybrid_backend_advances_provider_revision_hook():
    mapped = AgentBlenderStateMapper().map(abws_payload())
    writer = MemoryBackend(mapped)

    class Reader:
        revisions = 0

        def __call__(self):
            return writer.read_state().to_dict()

        def increment_revision(self):
            self.revisions += 1

    reader = Reader()
    backend = AgentBlenderWorldStateBackend(writer, reader)
    before = backend.read_state()
    backend.apply(
        SemanticAction("revision-1", ActionType.MOVE_OBJECT, "Cup", parameters={"location": [1, 2, 3]})
    )
    backend.restore(before)
    assert reader.revisions == 2


def test_maps_semantic_action_to_transaction_with_stable_ids():
    state = AgentBlenderStateMapper().map(abws_payload())
    action = SemanticAction(
        "move-1",
        ActionType.PLACE_ON,
        "Cup",
        "Table",
        parameters={"location": [0, 0, 1.71]},
        expected_effects=[Constraint("Cup", "on_top_of", "Table")],
    )
    plan = AgentBlenderActionMapper().map(action, state)
    assert plan["scene_revision"] == 7
    assert plan["plan"]["operations"][0]["target"] == "cup-uuid"
    assert plan["plan"]["operations"][0]["parameters"]["support"] == "table-uuid"
    assert plan["write_set"] == ["objects.cup-uuid.transform.location"]
    assert plan["derived_set"] == ["relations*"]


def test_transaction_backend_commits_and_undoes():
    payload = abws_payload()
    calls = []

    def execute(plan):
        calls.append(plan)
        payload["revision"] += 1
        operation = plan["plan"]["operations"][0]
        payload["objects"][1]["transform"]["position"] = operation["parameters"]["location"]
        return {"status": "committed", "transaction_id": "tx-1", "diff": {"Cup.location": "changed"}}

    undone = []
    backend = AgentBlenderTransactionBackend(
        lambda: payload,
        execute,
        undo_transaction=lambda transaction_id: undone.append(transaction_id) or {"success": True},
    )
    record = SafeExecutor(backend, solve_constraints=False).execute(
        SemanticAction("move-1", ActionType.MOVE_OBJECT, "Cup", parameters={"location": [1, 2, 3]})
    )
    assert record.status == "success"
    assert backend.read_state().timestamp == 8
    assert calls[0]["scene_revision"] == 7
    backend.restore(record.before)
    assert undone == ["tx-1"]


def test_transaction_rejection_reaches_failure_attribution():
    backend = AgentBlenderTransactionBackend(
        abws_payload,
        lambda plan: {
            "status": "rejected",
            "message": "write guard failed",
            "failed_constraint": "no_intersection",
            "property_diff": {"Cup.location": {"before": [0, 0, 1.7], "after": [2, 0, 1.7]}},
        },
    )
    record = SafeExecutor(backend, solve_constraints=False).execute(
        SemanticAction("bad-1", ActionType.MOVE_OBJECT, "Cup", parameters={"location": [2, 0, 1.7]})
    )
    failure = FailureAnalyzer().attribute(record, VerificationReport(False, "replan"))
    assert record.status == "failed"
    assert record.error_details["source"] == "agentblender_world_state"
    assert failure.failure_type == "agentblender_transaction_rejected"
    assert failure.failed_constraint == "no_intersection"
    assert "Cup.location" in failure.evidence["property_diff"]


def test_runtime_bridge_binds_state_execute_and_history():
    class Runtime:
        def __init__(self):
            self.current_state = abws_payload()
            self.plans = []
            self.undone = []
            self.redone = []

        def preview_and_commit(self, plan):
            self.plans.append(plan)
            self.current_state["revision"] += 1
            return {"committed": True, "transaction_id": "tx-bridge"}

        def undo(self, transaction_id):
            self.undone.append(transaction_id)
            return {"success": True}

        def redo(self, transaction_id):
            self.redone.append(transaction_id)
            return {"accepted": True}

    runtime = Runtime()
    bridge = AgentBlenderRuntimeBridge(runtime)
    backend = bridge.backend()
    before = backend.read_state()
    backend.apply(
        SemanticAction("bridge-1", ActionType.MOVE_OBJECT, "Cup", parameters={"location": [2, 0, 2]})
    )
    assert runtime.plans[0]["scene_revision"] == 7
    assert backend.read_state().timestamp == 8
    backend.restore(before)
    backend.redo()
    assert runtime.undone == ["tx-bridge"]
    assert runtime.redone == [None]


def test_runtime_bridge_coerces_annotated_pydantic_style_plan():
    class Plan:
        def __init__(self, payload):
            self.payload = payload

        @classmethod
        def model_validate(cls, payload):
            return cls(payload)

    class Runtime:
        state = abws_payload()

        def execute_plan(self, plan: Plan):
            assert isinstance(plan, Plan)
            return {"status": "committed", "id": "tx-model"}

    backend = AgentBlenderRuntimeBridge(Runtime()).backend()
    backend.apply(
        SemanticAction("model-1", ActionType.MOVE_OBJECT, "Cup", parameters={"location": [2, 0, 2]})
    )
    assert backend.last_transaction["id"] == "tx-model"
