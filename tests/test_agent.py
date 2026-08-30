from verigraph3d.agent import VeriGraph3DAgent
from verigraph3d.backends import MemoryBackend
from verigraph3d.cli import demo_scene


def test_end_to_end_natural_language_task():
    result = VeriGraph3DAgent(MemoryBackend(demo_scene())).run(instruction="把Cup放到Table上，并把Cup改成红色")
    assert result["accepted"]
    assert result["final_state"]["objects"]["Cup"]["color"] == (1.0, 0.0, 0.0, 1.0)
    assert result["metrics"]["actions"] == 2

