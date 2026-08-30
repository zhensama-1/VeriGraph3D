from pathlib import Path
import shutil

from PIL import Image

from verigraph3d.agent import AgentConfig, VeriGraph3DAgent
from verigraph3d.backends import MemoryBackend
from verigraph3d.cli import demo_scene
from verigraph3d.models import ActionType, GoalSpec, SemanticAction
from verigraph3d.visual import ReferenceImageVerifier


def image(path: Path, color: tuple[int, int, int]) -> str:
    Image.new("RGB", (64, 64), color).save(path)
    return str(path)


def test_reference_image_verifier_distinguishes_match_and_mismatch(tmp_path):
    reference = image(tmp_path / "reference.png", (255, 0, 0))
    matching = image(tmp_path / "matching.png", (255, 0, 0))
    mismatch = image(tmp_path / "mismatch.png", (0, 0, 255))
    verifier = ReferenceImageVerifier(size=64)
    state = demo_scene()
    state.metadata["render_path"] = matching
    match_score, _, _ = verifier.verify(state, GoalSpec(), [reference])
    state.metadata["render_path"] = mismatch
    mismatch_score, differences, _ = verifier.verify(state, GoalSpec(), [reference])
    assert match_score > 0.99
    assert mismatch_score < 0.8
    assert differences


class RenderableMemoryBackend(MemoryBackend):
    def __init__(self, reference: str):
        super().__init__(demo_scene())
        self.reference = reference

    def render(self, path: str, camera_name: str | None = None) -> str:
        shutil.copyfile(self.reference, path)
        return path


def test_agent_renders_after_action_and_counts_cost(tmp_path):
    reference = image(tmp_path / "reference.png", (255, 0, 0))
    backend = RenderableMemoryBackend(reference)
    config = AgentConfig(render_after_action=True, render_directory=str(tmp_path / "renders"))
    action = SemanticAction("color_001", ActionType.SET_MATERIAL, "Cup", parameters={"color": (1, 0, 0, 1)})
    result = VeriGraph3DAgent(backend, config=config, visual_verifier=ReferenceImageVerifier()).run(
        goal=GoalSpec(), actions=[action], reference_images=[reference]
    )
    assert result["accepted"]
    assert result["metrics"]["renders"] == 1
