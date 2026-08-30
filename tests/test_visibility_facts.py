from verigraph3d.models import CameraState, SceneObject, SceneState
from verigraph3d.state import RelationCalculator


def test_engine_visibility_score_overrides_fov_proxy_and_preserves_provenance():
    cup = SceneObject("Cup", metadata={"camera_visibility": {"Camera": 0.0}})
    state = SceneState(objects={"Cup": cup}, cameras={"Camera": CameraState()})
    fact = next(f for f in RelationCalculator().facts(state) if f.predicate == "visible_from")
    assert not fact.value
    assert fact.confidence == 0.0
    assert fact.source == "engine_ray_cast"
