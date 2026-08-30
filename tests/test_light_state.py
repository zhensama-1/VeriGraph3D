from verigraph3d.backends import MemoryBackend
from verigraph3d.execution import SafeExecutor
from verigraph3d.graph import ExecutableSceneGraph
from verigraph3d.models import ActionType, Constraint, GoalSpec, LightState, SceneState, SemanticAction
from verigraph3d.verification import HybridVerifier


def test_light_is_executable_state_and_graph_node():
    backend = MemoryBackend(SceneState(lights={"Key": LightState("Key")}))
    action = SemanticAction("light_001", ActionType.SET_LIGHT, "Key", parameters={"energy": 500.0, "color": (1.0, 0.5, 0.25)})
    record = SafeExecutor(backend).execute(action)
    assert record.status == "success"
    state = backend.read_state()
    assert state.lights["Key"].energy == 500.0
    assert record.changes["Key"]["entity_type"] == "light"
    graph = ExecutableSceneGraph()
    graph.rebuild(state)
    assert graph.nodes["Key"]["type"] == "light"
    report = HybridVerifier().verify(state, GoalSpec(required=[Constraint("Key", "energy", expected=500.0)]))
    assert report.accepted
