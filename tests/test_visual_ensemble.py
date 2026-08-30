from verigraph3d.cli import demo_scene
from verigraph3d.models import GoalSpec
from verigraph3d.visual import EnsembleVisualVerifier


class FixedVerifier:
    def __init__(self, score, difference, recommendation):
        self.result = score, [difference], [recommendation]

    def verify(self, state, goal, reference_images):
        return self.result


def test_visual_ensemble_combines_scores_and_evidence():
    verifier = EnsembleVisualVerifier(
        [FixedVerifier(0.5, "pixel mismatch", "move_camera"), FixedVerifier(1.0, "semantic ok", "none")],
        [0.4, 0.6],
    )
    score, differences, recommendations = verifier.verify(demo_scene(), GoalSpec(), [])
    assert score == 0.8
    assert differences == ["pixel mismatch", "semantic ok"]
    assert recommendations == ["move_camera", "none"]
