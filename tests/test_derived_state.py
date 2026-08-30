from verigraph3d.models import CameraState, GoalSpec, SceneObject, SceneState
from verigraph3d.verification import DeterministicVerifier


def test_visibility_metadata_change_is_not_an_object_modification():
    before = SceneState(
        objects={"Table": SceneObject("Table", metadata={"camera_visibility": {"Camera": 0.5}})},
        cameras={"Camera": CameraState()},
    )
    after = before.clone()
    after.objects["Table"].metadata["camera_visibility"]["Camera"] = 0.8
    results = DeterministicVerifier().verify(after, GoalSpec(), before, set())
    unexpected = next(result for result in results if result.code == "unexpected_modification")
    assert unexpected.passed
