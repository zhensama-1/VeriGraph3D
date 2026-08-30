from verigraph3d.graph import ExecutableSceneGraph
from verigraph3d.models import CameraState, Constraint, SceneState


def test_scalar_numeric_constraint_uses_tolerance() -> None:
    graph = ExecutableSceneGraph()
    graph.rebuild(
        SceneState(cameras={"Camera": CameraState(name="Camera", fov_degrees=60.0000017)})
    )

    assert graph.check(
        Constraint("Camera", "fov_degrees", expected=60.0, tolerance=0.001)
    )
    assert not graph.check(
        Constraint("Camera", "fov_degrees", expected=61.0, tolerance=0.001)
    )
