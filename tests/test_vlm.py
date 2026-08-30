from pathlib import Path

import pytest

from verigraph3d.agent import VeriGraph3DAgent
from verigraph3d.backends import MemoryBackend
from verigraph3d.cli import demo_scene
from verigraph3d.models import GoalSpec
from verigraph3d.vlm import (
    ChatCompletionsVLMClient, VLMError, VLMImage, VLMRequest, VLMResponse,
    VLMSettings, VLMTaskInterpreter, VLMUsage, VLMVisualVerifier, task_from_vlm_data,
)


def constraint(subject, predicate, obj=None, expected=True):
    return {
        "subject": subject, "predicate": predicate, "object": obj, "expected": expected,
        "hard": True, "tolerance": 0.02, "weight": 1.0,
    }


class FakeVLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.usage = VLMUsage()
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        self.usage.calls += 1
        self.usage.input_tokens += 10
        self.usage.output_tokens += 5
        data = self.responses.pop(0)
        return VLMResponse("{}", data, "fake-vlm", 10, 5)


def test_vlm_task_interpreter_drives_agent_and_records_usage():
    effect = constraint("Cup", "on_top_of", "Table")
    response = {
        "required": [effect], "forbidden": [constraint("Cup", "intersecting", "Table")], "visual": {},
        "actions": [{
            "id": "action_001", "type": "place_on", "target": "Cup", "reference": "Table",
            "parameters": {}, "preconditions": [], "expected_effects": [effect], "rollback_strategy": "restore_snapshot",
        }],
    }
    client = FakeVLMClient([response])
    agent = VeriGraph3DAgent(MemoryBackend(demo_scene()), task_interpreter=VLMTaskInterpreter(client))
    result = agent.run(instruction="把杯子放到桌子上")
    assert result["accepted"]
    assert result["metrics"]["vlm_calls"] == 1
    assert result["metrics"]["input_tokens"] == 10
    assert client.requests[0].json_schema


def test_vlm_rejects_action_on_unknown_entity():
    data = {
        "required": [], "forbidden": [], "visual": {},
        "actions": [{"id": "x", "type": "move_object", "target": "Ghost", "parameters": {"location": [0, 0, 0]}}],
    }
    with pytest.raises(VLMError, match="unknown entity"):
        task_from_vlm_data(data, demo_scene())


def test_vlm_rejects_goal_on_unknown_entity():
    data = {"required": [constraint("Ghost", "on_top_of", "Table")], "forbidden": [], "visual": {}, "actions": []}
    with pytest.raises(VLMError, match="goal targets unknown entity"):
        task_from_vlm_data(data, demo_scene())


def test_vlm_visual_verifier_uses_render_and_references(tmp_path):
    render = tmp_path / "render.png"
    reference = tmp_path / "reference.png"
    render.write_bytes(b"image")
    reference.write_bytes(b"image")
    client = FakeVLMClient([{"score": 0.91, "differences": ["minor camera offset"], "recommendations": ["adjust_camera"]}])
    state = demo_scene()
    state.metadata["render_path"] = str(render)
    score, differences, recommendations = VLMVisualVerifier(client).verify(state, GoalSpec(), [str(reference)])
    assert score == 0.91
    assert differences == ["minor camera offset"]
    assert recommendations == ["adjust_camera"]
    assert len(client.requests[0].images) == 2


def test_chat_compatible_payload_embeds_local_image(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"small image")
    client = ChatCompletionsVLMClient(VLMSettings("chat", "http://localhost:8000/v1", "local-vlm", ""))
    payload = client._payload(VLMRequest("system", "prompt", (VLMImage(str(image)),)))
    url = payload["messages"][1]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
