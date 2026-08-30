from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .models import (
    ActionType, CameraState, Constraint, GoalSpec, LightState, SceneObject,
    SceneState, SemanticAction,
)


SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class TaskCase:
    id: str
    instruction: str
    initial_scene: str | dict[str, Any]
    goal: GoalSpec
    seed: int = 42
    reference_images: list[str] = field(default_factory=list)
    uncertainties: list[dict[str, Any]] = field(default_factory=list)
    actions: list[SemanticAction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskDataset:
    def __init__(self, tasks: list[TaskCase], scenes: dict[str, SceneState] | None = None) -> None:
        self.tasks = tasks
        self.scenes = scenes or {}

    @classmethod
    def load(cls, path: str | Path) -> TaskDataset:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            raw_tasks, raw_scenes = payload, {}
        else:
            if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
                raise ValueError(f"Unsupported dataset schema: {payload.get('schema_version')}")
            raw_tasks, raw_scenes = payload.get("tasks", []), payload.get("scenes", {})
        scenes = {name: scene_from_dict(value) for name, value in raw_scenes.items()}
        tasks = [task_from_dict(item) for item in raw_tasks]
        for task in tasks:
            task.reference_images = [
                str(image if image.is_absolute() else (source.parent / image).resolve())
                for raw in task.reference_images
                for image in [Path(raw)]
            ]
        if len({task.id for task in tasks}) != len(tasks):
            raise ValueError("Task IDs must be unique")
        return cls(tasks, scenes)

    def initial_state(self, task: TaskCase) -> SceneState:
        if isinstance(task.initial_scene, dict):
            return scene_from_dict(task.initial_scene)
        if task.initial_scene not in self.scenes:
            raise KeyError(f"Scene '{task.initial_scene}' is not defined in the dataset")
        return self.scenes[task.initial_scene].clone()


def constraint_from_value(value: list | dict, forbidden: bool = False) -> Constraint:
    if isinstance(value, list):
        if len(value) != 3:
            raise ValueError(f"Constraint triples require 3 values: {value}")
        return Constraint(str(value[0]), str(value[1]), value[2], expected=True)
    return Constraint(**value)


def task_from_dict(value: dict[str, Any]) -> TaskCase:
    goal = GoalSpec(
        required=[constraint_from_value(c) for c in value.get("required", [])],
        forbidden=[constraint_from_value(c, True) for c in value.get("forbidden", [])],
        visual=value.get("visual", {}),
    )
    return TaskCase(
        id=value["id"], instruction=value["instruction"], initial_scene=value["initial_scene"],
        goal=goal, seed=int(value.get("seed", 42)), reference_images=value.get("reference_images", []),
        uncertainties=value.get("uncertainties", []),
        actions=[action_from_dict(item, index) for index, item in enumerate(value.get("actions", []), 1)],
        metadata=value.get("metadata", {}),
    )


def action_from_dict(value: dict[str, Any], index: int = 1) -> SemanticAction:
    return SemanticAction(
        id=str(value.get("id", f"action_{index:03d}")),
        type=ActionType(value["type"]), target=str(value["target"]),
        reference=value.get("reference"), parameters=dict(value.get("parameters", {})),
        preconditions=[constraint_from_value(item) for item in value.get("preconditions", [])],
        expected_effects=[constraint_from_value(item) for item in value.get("expected_effects", [])],
        rollback_strategy=str(value.get("rollback_strategy", "restore_snapshot")),
    )


def scene_from_dict(value: dict[str, Any]) -> SceneState:
    vector_fields = {"location", "rotation", "scale", "dimensions", "color"}
    objects = {}
    for name, attributes in value.get("objects", {}).items():
        normalized = {key: tuple(item) if key in vector_fields else item for key, item in attributes.items()}
        objects[name] = SceneObject(name=name, **normalized)
    cameras = {}
    for name, attributes in value.get("cameras", {}).items():
        normalized = {key: tuple(item) if key in {"location", "target"} else item for key, item in attributes.items()}
        cameras[name] = CameraState(name=name, **normalized)
    lights = {}
    for name, attributes in value.get("lights", {}).items():
        normalized = {key: tuple(item) if key in {"location", "color"} else item for key, item in attributes.items()}
        lights[name] = LightState(name=name, **normalized)
    return SceneState(objects=objects, cameras=cameras, lights=lights, metadata=value.get("metadata", {}))
