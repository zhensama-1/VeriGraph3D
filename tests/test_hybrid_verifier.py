from verigraph3d.models import Constraint, GoalSpec, SceneObject, SceneState
from verigraph3d.verification import HybridVerifier


def test_hard_geometry_failure_overrides_visual_success():
    state = SceneState(objects={
        "Table": SceneObject("Table", location=(0, 0, 0.5), dimensions=(4, 3, 1)),
        "Cup": SceneObject("Cup", location=(0, 0, 0.75), dimensions=(0.5, 0.5, 0.5)),
    })
    goal = GoalSpec(forbidden=[Constraint("Cup", "intersecting", "Table")])
    report = HybridVerifier().verify(state, goal)
    assert not report.accepted
    assert report.decision in {"local_repair", "replan"}

