from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
import json
import mimetypes
import os
from pathlib import Path
import re
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import ActionType, Constraint, GoalSpec, SceneState, SemanticAction


class VLMError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VLMImage:
    source: str
    detail: str = "auto"

    def as_url(self, max_bytes: int = 20 * 1024 * 1024) -> str:
        if self.source.startswith(("https://", "http://", "data:")):
            return self.source
        path = Path(self.source)
        if not path.is_file():
            raise VLMError(f"Image not found: {path}")
        size = path.stat().st_size
        if size > max_bytes:
            raise VLMError(f"Image exceeds {max_bytes} bytes: {path}")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            raise VLMError(f"Unsupported image type: {mime}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"


@dataclass(frozen=True, slots=True)
class VLMRequest:
    system: str
    prompt: str
    images: tuple[VLMImage, ...] = ()
    json_schema: dict[str, Any] | None = None
    schema_name: str = "verigraph3d_output"
    temperature: float = 0.0
    max_output_tokens: int = 2048


@dataclass(slots=True)
class VLMUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class VLMResponse:
    text: str
    data: dict[str, Any]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class VLMClient(Protocol):
    usage: VLMUsage

    def complete(self, request: VLMRequest) -> VLMResponse: ...


@dataclass(frozen=True, slots=True)
class VLMSettings:
    provider: str
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float = 120.0
    structured_outputs: bool = True
    max_retries: int = 2
    retry_delay_seconds: float = 0.5

    @classmethod
    def from_env(cls, prefix: str = "VERIGRAPH_VLM_") -> VLMSettings:
        provider = os.getenv(f"{prefix}PROVIDER", "responses").lower()
        base_url = os.getenv(f"{prefix}BASE_URL", "https://api.openai.com/v1")
        model = os.getenv(f"{prefix}MODEL", "")
        api_key = os.getenv(f"{prefix}API_KEY", "")
        if not model:
            raise VLMError(f"{prefix}MODEL is required")
        if not api_key and not base_url.startswith(("http://localhost", "http://127.0.0.1")):
            raise VLMError(f"{prefix}API_KEY is required for remote endpoints")
        structured = os.getenv(f"{prefix}STRUCTURED_OUTPUTS", "true").lower() not in {"0", "false", "no"}
        return cls(
            provider, base_url, model, api_key,
            float(os.getenv(f"{prefix}TIMEOUT", "120")), structured,
            int(os.getenv(f"{prefix}MAX_RETRIES", "2")),
            float(os.getenv(f"{prefix}RETRY_DELAY", "0.5")),
        )


class _HTTPVLMClient:
    endpoint = ""

    def __init__(self, settings: VLMSettings) -> None:
        self.settings = settings
        self.usage = VLMUsage()

    def complete(self, request: VLMRequest) -> VLMResponse:
        payload = self._payload(request)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "User-Agent": "VeriGraph3D/0.1"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        http_request = Request(f"{self.settings.base_url.rstrip('/')}/{self.endpoint}", data=body, headers=headers, method="POST")
        raw = None
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            self.usage.calls += 1
            try:
                with urlopen(http_request, timeout=self.settings.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                message = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = VLMError(f"VLM HTTP {exc.code}: {message}")
                if exc.code not in {408, 409, 429} and exc.code < 500:
                    raise last_error from exc
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = VLMError(f"VLM request failed: {exc}")
            if attempt < self.settings.max_retries:
                time.sleep(self.settings.retry_delay_seconds * (2**attempt))
        if raw is None:
            raise last_error or VLMError("VLM request failed without a response")
        text, model, input_tokens, output_tokens = self._parse(raw)
        data = parse_json_object(text)
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        return VLMResponse(text, data, model, input_tokens, output_tokens)

    def _payload(self, request: VLMRequest) -> dict[str, Any]:
        raise NotImplementedError

    def _parse(self, raw: dict[str, Any]) -> tuple[str, str, int, int]:
        raise NotImplementedError


class ResponsesVLMClient(_HTTPVLMClient):
    endpoint = "responses"

    def _payload(self, request: VLMRequest) -> dict[str, Any]:
        prompt = _prompt_with_schema(request) if request.json_schema and not self.settings.structured_outputs else request.prompt
        content = [{"type": "input_text", "text": prompt}]
        content.extend({"type": "input_image", "image_url": image.as_url(), "detail": image.detail} for image in request.images)
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "instructions": request.system,
            "input": [{"role": "user", "content": content}],
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
            "store": False,
        }
        if request.json_schema and self.settings.structured_outputs:
            payload["text"] = {"format": {"type": "json_schema", "name": request.schema_name, "schema": request.json_schema, "strict": True}}
        return payload

    def _parse(self, raw: dict[str, Any]) -> tuple[str, str, int, int]:
        text = raw.get("output_text", "")
        if not text:
            parts = [
                part.get("text", "")
                for item in raw.get("output", []) if item.get("type") == "message"
                for part in item.get("content", []) if part.get("type") == "output_text"
            ]
            text = "".join(parts)
        if not text:
            raise VLMError("Responses API returned no output text")
        usage = raw.get("usage", {})
        return text, raw.get("model", self.settings.model), int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


class ChatCompletionsVLMClient(_HTTPVLMClient):
    endpoint = "chat/completions"

    def _payload(self, request: VLMRequest) -> dict[str, Any]:
        prompt = _prompt_with_schema(request) if request.json_schema and not self.settings.structured_outputs else request.prompt
        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": image.as_url(), "detail": image.detail}} for image in request.images)
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [{"role": "system", "content": request.system}, {"role": "user", "content": content}],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if request.json_schema and self.settings.structured_outputs:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": request.schema_name, "schema": request.json_schema, "strict": True}}
        return payload

    def _parse(self, raw: dict[str, Any]) -> tuple[str, str, int, int]:
        try:
            text = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VLMError("Chat Completions API returned no message content") from exc
        usage = raw.get("usage", {})
        return str(text), raw.get("model", self.settings.model), int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def create_vlm_client(settings: VLMSettings | None = None) -> VLMClient:
    settings = settings or VLMSettings.from_env()
    if settings.provider == "responses":
        return ResponsesVLMClient(settings)
    if settings.provider in {"chat", "chat_completions", "openai_compatible"}:
        return ChatCompletionsVLMClient(settings)
    raise VLMError(f"Unsupported VLM provider: {settings.provider}")


def _prompt_with_schema(request: VLMRequest) -> str:
    return request.prompt + "\n\nReturn only a JSON object matching this schema:\n" + json.dumps(request.json_schema, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class VLMBudget:
    max_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None


class BudgetedVLMClient:
    """Enforces per-agent logical budgets around any VLMClient."""

    def __init__(self, client: VLMClient, budget: VLMBudget) -> None:
        self.client = client
        self.budget = budget
        self.usage = client.usage

    def complete(self, request: VLMRequest) -> VLMResponse:
        self._require_remaining()
        response = self.client.complete(request)
        self._require_remaining(after_response=True)
        return response

    def _require_remaining(self, after_response: bool = False) -> None:
        checks = (
            (self.budget.max_calls, self.usage.calls, "calls"),
            (self.budget.max_input_tokens, self.usage.input_tokens, "input tokens"),
            (self.budget.max_output_tokens, self.usage.output_tokens, "output tokens"),
        )
        for limit, used, label in checks:
            exceeded = used > limit if after_response else used >= limit
            if limit is not None and exceeded:
                raise VLMError(f"VLM budget exhausted: {label} {used}/{limit}")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise VLMError("VLM output is not a JSON object")
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise VLMError(f"Invalid JSON returned by VLM: {exc}") from exc
    if not isinstance(value, dict):
        raise VLMError("VLM output must be a JSON object")
    return value


TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "required": {"type": "array", "items": {"$ref": "#/$defs/constraint"}},
        "forbidden": {"type": "array", "items": {"$ref": "#/$defs/constraint"}},
        "visual": {"type": "object", "additionalProperties": True},
        "actions": {"type": "array", "items": {"$ref": "#/$defs/action"}},
    },
    "required": ["required", "forbidden", "visual", "actions"],
    "additionalProperties": False,
    "$defs": {
        "constraint": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"}, "predicate": {"type": "string"},
                "object": {"type": ["string", "null"]}, "expected": {},
                "hard": {"type": "boolean"}, "tolerance": {"type": "number"}, "weight": {"type": "number"},
            },
            "required": ["subject", "predicate", "object", "expected", "hard", "tolerance", "weight"],
            "additionalProperties": False,
        },
        "action": {
            "type": "object",
            "properties": {
                "id": {"type": "string"}, "type": {"type": "string", "enum": [item.value for item in ActionType]},
                "target": {"type": "string"}, "reference": {"type": ["string", "null"]},
                "parameters": {"type": "object", "additionalProperties": True},
                "preconditions": {"type": "array", "items": {"$ref": "#/$defs/constraint"}},
                "expected_effects": {"type": "array", "items": {"$ref": "#/$defs/constraint"}},
                "rollback_strategy": {"type": "string"},
            },
            "required": ["id", "type", "target", "reference", "parameters", "preconditions", "expected_effects", "rollback_strategy"],
            "additionalProperties": False,
        },
    },
}


VISUAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "differences": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "differences", "recommendations"],
    "additionalProperties": False,
}


class VLMTaskInterpreter:
    def __init__(self, client: VLMClient) -> None:
        self.client = client

    def parse(self, instruction: str, state: SceneState, reference_images: list[str] | None = None) -> tuple[GoalSpec, list[SemanticAction]]:
        scene = {
            "objects": {name: {key: value for key, value in asdict(obj).items() if key != "metadata"} for name, obj in state.objects.items()},
            "cameras": {name: asdict(camera) for name, camera in state.cameras.items()},
            "lights": {name: asdict(light) for name, light in state.lights.items()},
        }
        prompt = "Instruction:\n" + instruction + "\n\nCurrent deterministic scene state:\n" + json.dumps(scene, ensure_ascii=False)
        response = self.client.complete(VLMRequest(
            system=(
                "Convert a 3D editing request into the supplied JSON schema. Use only listed scene entity names unless creating an object. "
                "Actions must use the allowed semantic action types; never output code. Constraints are the authoritative completion criteria."
            ),
            prompt=prompt,
            images=tuple(VLMImage(path) for path in reference_images or []),
            json_schema=TASK_SCHEMA,
            schema_name="verigraph3d_task",
        ))
        return task_from_vlm_data(response.data, state)


class VLMVisualVerifier:
    def __init__(self, client: VLMClient) -> None:
        self.client = client

    def verify(self, state: SceneState, goal: GoalSpec, reference_images: list[str]):
        render_path = state.metadata.get("render_path")
        if not render_path:
            return 0.0, ["candidate render is missing"], ["render_scene"]
        images = [VLMImage(str(render_path)), *(VLMImage(path) for path in reference_images)]
        prompt = (
            "The first image is the current Blender render; remaining images are references. "
            "Evaluate composition, object appearance, color/material, spatial layout, camera, and occlusion. "
            "Return a strict score in [0,1], concrete differences, and actionable semantic repairs.\nGoal:\n"
            + json.dumps(asdict(goal), ensure_ascii=False)
        )
        response = self.client.complete(VLMRequest(
            system="You are the visual verification component of a safe 3D editing agent. Do not propose arbitrary code.",
            prompt=prompt, images=tuple(images), json_schema=VISUAL_SCHEMA, schema_name="verigraph3d_visual_report",
        ))
        score = max(0.0, min(1.0, float(response.data["score"])))
        return score, [str(item) for item in response.data["differences"]], [str(item) for item in response.data["recommendations"]]


def _constraint(value: dict[str, Any]) -> Constraint:
    return Constraint(
        subject=str(value["subject"]), predicate=str(value["predicate"]), object=value.get("object"),
        expected=value.get("expected", True), hard=bool(value.get("hard", True)),
        tolerance=float(value.get("tolerance", 0.01)), weight=float(value.get("weight", 1.0)),
    )


def task_from_vlm_data(data: dict[str, Any], state: SceneState) -> tuple[GoalSpec, list[SemanticAction]]:
    goal = GoalSpec(
        required=[_constraint(value) for value in data.get("required", [])],
        forbidden=[_constraint(value) for value in data.get("forbidden", [])],
        visual=dict(data.get("visual", {})),
    )
    actions = []
    allowed_entities = set(state.objects) | set(state.cameras) | set(state.lights)
    allowed_entities.update(
        str(value["target"])
        for value in data.get("actions", [])
        if value.get("type") == ActionType.CREATE_OBJECT.value and value.get("target")
    )
    for constraint in [*goal.required, *goal.forbidden]:
        if constraint.subject not in allowed_entities:
            raise VLMError(f"VLM goal targets unknown entity: {constraint.subject}")
        if constraint.object is not None and constraint.object not in allowed_entities:
            raise VLMError(f"VLM goal references unknown entity: {constraint.object}")
    action_ids: set[str] = set()
    for index, value in enumerate(data.get("actions", []), 1):
        action_type = ActionType(value["type"])
        target = str(value["target"])
        if action_type != ActionType.CREATE_OBJECT and target not in allowed_entities:
            raise VLMError(f"VLM action targets unknown entity: {target}")
        reference = value.get("reference")
        if reference is not None and reference not in allowed_entities:
            raise VLMError(f"VLM action references unknown entity: {reference}")
        action_id = str(value.get("id") or f"action_{index:03d}")
        if action_id in action_ids:
            raise VLMError(f"VLM returned duplicate action ID: {action_id}")
        action_ids.add(action_id)
        actions.append(SemanticAction(
            id=action_id, type=action_type, target=target,
            reference=reference, parameters=dict(value.get("parameters", {})),
            preconditions=[_constraint(item) for item in value.get("preconditions", [])],
            expected_effects=[_constraint(item) for item in value.get("expected_effects", [])],
            rollback_strategy=str(value.get("rollback_strategy", "restore_snapshot")),
        ))
    return goal, actions
