from verigraph3d.backends import MemoryBackend
from verigraph3d.execution import SafeExecutor
from verigraph3d.models import ActionType, SceneObject, SceneState, SemanticAction
from verigraph3d.solver import PlanPreflight
from verigraph3d.state import RelationCalculator


def scene():
    return SceneState(objects={
        "Table": SceneObject("Table", location=(0, 0, 0.5), dimensions=(4, 3, 1)),
        "Cup": SceneObject("Cup", location=(3, 0, 0.25), dimensions=(0.5, 0.5, 0.5)),
    })


def test_place_on_and_rollback():
    backend = MemoryBackend(scene())
    executor = SafeExecutor(backend)
    record = executor.execute(SemanticAction("a1", ActionType.PLACE_ON, "Cup", "Table"))
    assert record.status == "success"
    facts = RelationCalculator(tolerance=0.02).facts(backend.read_state())
    assert any(f.subject == "Cup" and f.predicate == "on_top_of" and f.object == "Table" and f.value for f in facts)
    executor.rollback("a1")
    assert backend.read_state().objects["Cup"].location == (3, 0, 0.25)


def test_preflight_rejects_unknown_object():
    record = SafeExecutor(MemoryBackend(scene())).execute(SemanticAction("a2", ActionType.MOVE_OBJECT, "Ghost", parameters={"location": (0, 0, 0)}))
    assert record.status == "failed"
    assert "target_exists" in record.error


def test_create_preflight_rejects_duplicate_and_unlisted_primitive():
    state = SceneState(objects={"Existing": SceneObject("Existing")})
    preflight = PlanPreflight()
    duplicate = SemanticAction(
        "create_duplicate", ActionType.CREATE_OBJECT, "Existing",
        parameters={"primitive": "cube", "dimensions": [1, 1, 1]},
    )
    unsafe = SemanticAction(
        "create_unsafe", ActionType.CREATE_OBJECT, "New",
        parameters={"primitive": "custom_script", "dimensions": [1, 1, 1]},
    )

    assert any(not result.passed for result in preflight.check(duplicate, state))
    assert any(
        result.code == "allowed_primitive" and not result.passed
        for result in preflight.check(unsafe, state)
    )


def test_safe_create_parameters_execute_in_memory_backend():
    backend = MemoryBackend(SceneState())
    record = SafeExecutor(backend).execute(SemanticAction(
        "create", ActionType.CREATE_OBJECT, "Vase",
        parameters={
            "primitive": "cylinder", "category": "vase",
            "dimensions": [0.4, 0.4, 0.8], "location": [0, 0, 0.4],
            "color": [0.1, 0.3, 0.8, 1.0],
        },
    ))

    assert record.status == "success"
    assert backend.read_state().objects["Vase"].category == "vase"
