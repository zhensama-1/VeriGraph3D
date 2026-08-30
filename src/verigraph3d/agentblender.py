from __future__ import annotations

"""Adapter for AgentBlender World State (ABWS).

ABWS remains the owner of stable IDs, revisions and transactional state.  This
module only translates its serialisable world state into VeriGraph3D's compact
``SceneState`` contract; scene edits continue through the configured backend.
No ABWS source code is copied into this project.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import re
import shutil
from typing import Any
import json

from .models import (
    ActionType, CameraState, LightState, SceneBackend, SceneObject, SceneState,
    SemanticAction,
)


class AgentBlenderIntegrationError(RuntimeError):
    """Raised when an ABWS payload or installation cannot be used safely."""


class AgentBlenderTransactionRejected(AgentBlenderIntegrationError):
    """An ABWS preview/commit gate rejected a candidate edit."""

    def __init__(self, message: str, evidence: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.evidence = {"source": "agentblender_world_state", **dict(evidence or {})}


def _plain(value: Any) -> Any:
    """Convert Pydantic/dataclass-like values to ordinary Python containers."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    value = _plain(value)
    if not isinstance(value, Mapping):
        raise AgentBlenderIntegrationError(f"Expected a mapping, got {type(value).__name__}")
    return value


def _vec(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    value = _plain(value)
    if isinstance(value, Mapping):
        value = [value.get("x", default[0]), value.get("y", default[1]), value.get("z", default[2])]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return default
    return float(value[0]), float(value[1]), float(value[2])


def _rgba(value: Any) -> tuple[float, float, float, float]:
    value = _plain(value)
    if isinstance(value, Mapping):
        value = value.get("base_color", value.get("color"))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return 0.8, 0.8, 0.8, 1.0
    return float(value[0]), float(value[1]), float(value[2]), float(value[3]) if len(value) > 3 else 1.0


def _items(value: Any) -> list[tuple[str | None, Mapping[str, Any]]]:
    value = _plain(value) or []
    if isinstance(value, Mapping):
        return [(str(key), _mapping(item)) for key, item in value.items()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [(None, _mapping(item)) for item in value]
    raise AgentBlenderIntegrationError("ABWS collection must be a mapping or a list")


@dataclass(slots=True)
class AgentBlenderStateMapper:
    """Map ABWS JSON/Pydantic world state to VeriGraph3D state.

    The mapper accepts both list- and ID-keyed collections.  It deliberately
    preserves the original record in ``metadata['abws']`` so future ABWS fields
    are not lost while the research schemas evolve.
    """

    strict: bool = True

    def map(self, payload: Any) -> SceneState:
        root = _mapping(payload)
        if isinstance(root.get("state"), Mapping):
            root = _mapping(root["state"])
        objects: dict[str, SceneObject] = {}
        id_to_name: dict[str, str] = {}
        raw_objects = _items(root.get("objects", root.get("entities", [])))
        for key, record in raw_objects:
            stable_id = str(record.get("id", record.get("object_id", key or "")))
            name = str(record.get(
                "blender_name",
                record.get("name", record.get("display_name", record.get("label", stable_id))),
            ))
            if not name:
                if self.strict:
                    raise AgentBlenderIntegrationError("ABWS object has neither name nor stable ID")
                continue
            id_to_name[stable_id] = name
            objects[name] = self._object(record, name, stable_id)
        for obj in objects.values():
            parent_id = obj.metadata.pop("_abws_parent_id", None)
            if parent_id:
                obj.parent = id_to_name.get(str(parent_id), str(parent_id))

        cameras = {
            camera.name: camera
            for key, record in _items(root.get("cameras", []))
            for camera in [self._camera(record, key)]
        }
        lights = {
            light.name: light
            for key, record in _items(root.get("lights", []))
            for light in [self._light(record, key)]
        }
        revision = root.get("revision", root.get("version", root.get("timestamp", 0)))
        metadata = {
            "source": "agentblender_world_state",
            "abws_revision": revision,
            "abws_scene_id": root.get("scene_id", root.get("id")),
            "abws_relations": _plain(root.get("relations", [])),
            "abws_constraints": _plain(root.get("constraints", [])),
            "abws_environment": _plain(root.get("environment", {})),
        }
        try:
            timestamp = int(revision)
        except (TypeError, ValueError):
            timestamp = 0
        return SceneState(objects=objects, cameras=cameras, lights=lights, timestamp=timestamp, metadata=metadata)

    def _object(self, record: Mapping[str, Any], name: str, stable_id: str) -> SceneObject:
        transform = _mapping(record.get("transform", {}))
        location = _vec(
            transform.get("translation", transform.get("position", transform.get("location", record.get("location")))),
            (0.0, 0.0, 0.0),
        )
        rotation = _vec(transform.get("rotation_euler", transform.get("rotation", record.get("rotation"))), (0.0, 0.0, 0.0))
        scale = _vec(transform.get("scale", record.get("scale")), (1.0, 1.0, 1.0))
        bbox = _plain(record.get("aabb", record.get("bbox", record.get("bounds"))))
        metadata: dict[str, Any] = {"abws_id": stable_id, "abws": dict(record)}
        dimensions = _vec(record.get("dimensions", record.get("size")), (1.0, 1.0, 1.0))
        scene_location = location
        if isinstance(bbox, Mapping):
            minimum = _vec(bbox.get("minimum", bbox.get("min")), location)
            maximum = _vec(bbox.get("maximum", bbox.get("max")), location)
            metadata["world_aabb"] = {"minimum": minimum, "maximum": maximum}
            dimensions = tuple(maximum[i] - minimum[i] for i in range(3))
        elif record.get("origin_policy") in {"BOTTOM_CENTER", "CENTER"}:
            half_x = dimensions[0] * scale[0] / 2
            half_y = dimensions[1] * scale[1] / 2
            height = dimensions[2] * scale[2]
            minimum_z = (
                location[2]
                if record.get("origin_policy") == "BOTTOM_CENTER"
                else location[2] - height / 2
            )
            metadata["world_aabb"] = {
                "minimum": (location[0] - half_x, location[1] - half_y, minimum_z),
                "maximum": (location[0] + half_x, location[1] + half_y, minimum_z + height),
            }
        if record.get("origin_policy") == "BOTTOM_CENTER":
            scene_location = (
                location[0], location[1], location[2] + dimensions[2] * scale[2] / 2
            )
        if "world_aabb" in metadata:
            metadata["aabb_origin"] = scene_location
        parent = record.get("parent", record.get("parent_id"))
        if parent:
            metadata["_abws_parent_id"] = parent
        material_value = _plain(record.get("material"))
        materials = _plain(record.get("materials", []))
        if material_value is None and isinstance(materials, Sequence) and materials:
            material_value = _plain(materials[0])
        material = None
        if isinstance(material_value, Mapping):
            material = material_value.get(
                "name", material_value.get("id", material_value.get("material_id"))
            )
        elif material_value is not None:
            material = str(material_value)
        pbr = material_value.get("pbr_parameters", {}) if isinstance(material_value, Mapping) else {}
        color = _rgba(record.get("color", pbr.get("diffuse_color", material_value)))
        object_metadata = _plain(record.get("metadata", {}))
        visible = record.get("visible", not record.get("hidden", False))
        if isinstance(object_metadata, Mapping):
            visible = object_metadata.get("visible", visible)
        return SceneObject(
            name=name,
            category=str(record.get("category", record.get("type", "object"))).lower(),
            location=scene_location,
            rotation=rotation,
            scale=scale,
            dimensions=dimensions,
            color=color,
            material=str(material) if material is not None else None,
            visible=bool(visible),
            rigid_body=bool(record.get("rigid_body", record.get("physics", {}).get("rigid_body", False)) if isinstance(record.get("physics", {}), Mapping) else record.get("rigid_body", False)),
            collision_enabled=bool(record.get("collision_enabled", True)),
            metadata=metadata,
        )

    @staticmethod
    def _camera(record: Mapping[str, Any], key: str | None) -> CameraState:
        transform = _mapping(record.get("transform", {}))
        return CameraState(
            name=str(record.get("blender_name", record.get("name", key or record.get("id", "Camera")))),
            location=_vec(transform.get("translation", transform.get("location", record.get("location"))), (0.0, 0.0, 0.0)),
            target=_vec(record.get("target"), (0.0, 0.0, 0.0)),
            fov_degrees=float(record.get("fov_degrees", record.get("fov", 50.0))),
            metadata={"abws_id": str(record.get("id", key or "")), "abws": dict(record)},
        )

    @staticmethod
    def _light(record: Mapping[str, Any], key: str | None) -> LightState:
        transform = _mapping(record.get("transform", {}))
        return LightState(
            name=str(record.get("blender_name", record.get("name", key or record.get("id", "Light")))),
            light_type=str(record.get("light_type", record.get("type", "POINT"))).upper(),
            location=_vec(transform.get("translation", transform.get("location", record.get("location"))), (0.0, 0.0, 3.0)),
            color=_rgba(record.get("color"))[:3],
            energy=float(record.get("energy", record.get("power", 1000.0))),
            visible=bool(record.get("visible", True)),
            metadata={"abws_id": str(record.get("id", key or "")), "abws": dict(record)},
        )


class AgentBlenderJsonReader:
    """Read an ABWS state snapshot atomically produced by its JSON store/API."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __call__(self) -> Any:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AgentBlenderIntegrationError(f"Cannot read ABWS state {self.path}: {exc}") from exc


class AgentBlenderInstalledReader:
    """Resolve a live extractor from an installed ABWS package.

    A callable can be supplied explicitly, which is the stable integration
    contract.  Automatic discovery is intentionally conservative and only
    accepts zero-argument ``extract_scene_state`` entry points.
    """

    _CANDIDATES = (
        ("agent_blender.blender_state", "extract_current_scene"),
        ("agent_blender.blender_extractor", "extract_scene_state"),
        ("agent_blender.blender.extractor", "extract_scene_state"),
        ("agent_blender.extractor", "extract_scene_state"),
    )

    def __init__(self, extractor: Callable[[], Any] | None = None) -> None:
        self.extractor = extractor

    def __call__(self) -> Any:
        extractor = self.extractor or self._discover()
        return extractor()

    def increment_revision(self) -> int | None:
        """Advance a live ABWS Blender revision when that API is available."""
        try:
            module = importlib.import_module("agent_blender.blender_state")
            increment = getattr(module, "increment_scene_revision")
        except (ImportError, AttributeError):
            return None
        return int(increment())

    @classmethod
    def _discover(cls) -> Callable[[], Any]:
        errors: list[str] = []
        for module_name, attribute in cls._CANDIDATES:
            try:
                module = importlib.import_module(module_name)
                candidate = getattr(module, attribute)
                if callable(candidate):
                    return candidate
            except (ImportError, AttributeError) as exc:
                errors.append(f"{module_name}.{attribute}: {exc}")
        raise AgentBlenderIntegrationError(
            "ABWS is not installed or exposes no supported extractor. "
            "Pass AgentBlenderInstalledReader(extractor=...) explicitly. " + "; ".join(errors)
        )


class AgentBlenderWorldStateBackend(SceneBackend):
    """Use ABWS for reads and an existing backend for safe scene mutations.

    This composition lets ABWS be the authoritative state observer while the
    current BlenderBackend retains VeriGraph3D's fixed action dispatch table,
    checkpoints, rendering and active-view methods.
    """

    def __init__(
        self,
        write_backend: SceneBackend,
        state_reader: Callable[[], Any],
        mapper: AgentBlenderStateMapper | None = None,
    ) -> None:
        self.write_backend = write_backend
        self.state_reader = state_reader
        self.mapper = mapper or AgentBlenderStateMapper()

    def read_state(self) -> SceneState:
        return self.mapper.map(self.state_reader())

    def apply(self, action: SemanticAction) -> None:
        self.write_backend.apply(action)
        increment = getattr(self.state_reader, "increment_revision", None)
        if callable(increment):
            increment()

    def restore(self, state: SceneState) -> None:
        self.write_backend.restore(state)
        increment = getattr(self.state_reader, "increment_revision", None)
        if callable(increment):
            increment()

    def __getattr__(self, name: str) -> Any:
        # Preserve render/checkpoint/score_view capabilities of BlenderBackend.
        return getattr(self.write_backend, name)


@dataclass(slots=True)
class AgentBlenderActionMapper:
    """Translate one solved VeriGraph3D action into an auditable ABWS plan.

    The resulting payload is intentionally JSON-only.  A concrete ABWS
    installation may pass a custom ``plan_builder`` to the transaction backend
    when its Pydantic schema uses different field names.
    """

    def map(self, action: SemanticAction, state: SceneState) -> dict[str, Any]:
        entity_root = "objects"
        if action.type == ActionType.SET_CAMERA:
            target = state.cameras.get(action.target)
            entity_root = "cameras"
        elif action.type == ActionType.SET_LIGHT:
            target = state.lights.get(action.target)
            entity_root = "lights"
        else:
            target = state.objects.get(action.target)
        reference = state.objects.get(action.reference) if action.reference else None
        target_id = target.metadata.get("abws_id", action.target) if target else action.target
        reference_id = (
            reference.metadata.get("abws_id", action.reference) if reference else action.reference
        )
        operation_type, parameters = self._operation(action, reference_id, target)
        operation = {
            "op_id": action.id,
            "type": operation_type,
            "target": target_id,
            "parameters": parameters,
        }
        revision = state.metadata.get("abws_revision", state.timestamp)
        read_set = [f"{entity_root}.{target_id}.*"]
        if reference_id is not None:
            read_set.append(f"objects.{reference_id}.*")
        return {
            "transaction_id": f"vg3d_{action.id}",
            "instruction": f"VeriGraph3D semantic action {action.type.value}",
            "scene_revision": revision,
            "read_set": read_set,
            "write_set": self._write_set(operation_type, entity_root, str(target_id), parameters),
            "derived_set": ["relations*"],
            "preconditions": [{
                "path": f"{entity_root}.{target_id}.id",
                "operator": (
                    "not_exists" if operation_type == "CREATE_OBJECT" else "exists"
                ),
            }],
            "postconditions": self._postconditions(
                operation_type, entity_root, str(target_id), parameters
            ),
            "plan": {
                "plan_id": action.id,
                "scene_revision": revision,
                "goal": f"VeriGraph3D {action.type.value}",
                "resolved_references": (
                    {action.reference: str(reference_id)}
                    if action.reference and reference_id else {}
                ),
                "operations": [operation],
                "constraints": [],
            },
        }

    @staticmethod
    def _operation(
        action: SemanticAction, reference_id: Any, target: Any
    ) -> tuple[str, dict[str, Any]]:
        parameters = dict(action.parameters)
        mapping = {
            ActionType.CREATE_OBJECT: "CREATE_OBJECT",
            ActionType.MOVE_OBJECT: "MOVE_OBJECT",
            ActionType.ROTATE_OBJECT: "ROTATE_OBJECT",
            ActionType.SCALE_OBJECT: "SCALE_OBJECT",
            ActionType.DELETE_OBJECT: "DELETE_OBJECT",
            ActionType.PLACE_ON: "PLACE_ON",
            ActionType.PLACE_INSIDE: "PLACE_INSIDE",
            ActionType.ALIGN: "ALIGN_OBJECT",
            ActionType.SET_MATERIAL: "REPLACE_MATERIAL",
            ActionType.SET_CAMERA: "SET_CAMERA",
            ActionType.SET_LIGHT: "SET_LIGHT",
        }
        if action.type not in mapping:
            raise AgentBlenderIntegrationError(
                f"ABWS does not support semantic action {action.type.value}"
            )
        operation_type = mapping[action.type]
        if operation_type == "CREATE_OBJECT":
            primitive = str(parameters.get("primitive", "cube")).lower()
            if primitive not in {"cube", "sphere", "cylinder", "cone", "plane"}:
                raise AgentBlenderIntegrationError(
                    f"ABWS creation does not allow primitive {primitive!r}"
                )
            dimensions = list(parameters.get("dimensions", (1.0, 1.0, 1.0)))
            if len(dimensions) != 3 or any(
                not isinstance(value, (int, float)) or value <= 0 for value in dimensions
            ):
                raise AgentBlenderIntegrationError(
                    "ABWS object creation requires three positive dimensions"
                )
            scale = list(parameters.get("scale", (1.0, 1.0, 1.0)))
            location = list(parameters.get(
                "location", (0.0, 0.0, dimensions[2] * scale[2] / 2)
            ))
            if len(location) != 3:
                raise AgentBlenderIntegrationError(
                    "ABWS object creation requires a three-dimensional location"
                )
            # VeriGraph locations use object centres; ABWS managed mesh locations
            # use bottom centres so support and containment checks remain stable.
            location[2] -= dimensions[2] * abs(scale[2]) / 2
            color = parameters.get("color")
            material = parameters.get("material") or f"{action.target}_material"
            materials = []
            if color is not None or parameters.get("material"):
                materials.append({
                    "slot": 0,
                    "material_id": material,
                    "pbr_parameters": (
                        {"diffuse_color": list(color)} if color is not None else {}
                    ),
                })
            parameters = {"object": {
                "id": action.target,
                "blender_name": f"ABWS__{action.target}",
                "category": str(parameters.get("category", "object")),
                "display_name": str(parameters.get("display_name", action.target)),
                "transform": {
                    "location": location,
                    "rotation_euler": list(parameters.get(
                        "rotation_euler", parameters.get("rotation", (0.0, 0.0, 0.0))
                    )),
                    "scale": scale,
                },
                "dimensions": dimensions,
                "origin_policy": "BOTTOM_CENTER",
                "materials": materials,
                "capabilities": [
                    "move", "rotate", "scale", "duplicate", "delete",
                    "replace_material", "set_visibility",
                ],
                "metadata": {
                    "created_by": "verigraph3d", "locked": False, "visible": True,
                    "extra": {"primitive": primitive},
                },
            }}
        raw_target = target.metadata.get("abws", {}) if target else {}
        if (
            "location" in parameters
            and isinstance(raw_target, Mapping)
            and raw_target.get("origin_policy") == "BOTTOM_CENTER"
        ):
            location = list(parameters["location"])
            location[2] -= target.dimensions[2] * target.scale[2] / 2
            parameters["location"] = location
        if operation_type == "ROTATE_OBJECT" and "rotation" in parameters:
            parameters["rotation_euler"] = parameters.pop("rotation")
        if operation_type == "PLACE_ON":
            parameters.pop("location", None)
            parameters["support"] = reference_id
        elif operation_type == "PLACE_INSIDE":
            parameters["container"] = reference_id
        elif operation_type == "ALIGN_OBJECT":
            parameters["reference"] = reference_id
        elif operation_type == "REPLACE_MATERIAL":
            material = parameters.get("material")
            if "color" in parameters:
                operation_type = "SET_MATERIAL_COLOR"
                parameters = {
                    "color": parameters["color"],
                    "slot": int(parameters.get("slot", 0)),
                    **({"material_id": material} if material else {}),
                }
            elif material:
                parameters = {"material_id": material, "slot": int(parameters.get("slot", 0))}
            else:
                raise AgentBlenderIntegrationError(
                    "ABWS material edits require a material name or color"
                )
        elif operation_type == "SET_CAMERA" and "rotation" in parameters:
            parameters["rotation_euler"] = parameters.pop("rotation")
        elif operation_type == "SET_LIGHT" and "rotation" in parameters:
            parameters["rotation_euler"] = parameters.pop("rotation")
        return operation_type, parameters

    @staticmethod
    def _write_set(
        operation_type: str, entity_root: str, target_id: str,
        parameters: Mapping[str, Any],
    ) -> list[str]:
        suffixes = {
            "MOVE_OBJECT": ["transform.location"],
            "ROTATE_OBJECT": ["transform.rotation_euler"],
            "SCALE_OBJECT": ["transform.scale"],
            "PLACE_ON": ["transform.location"],
            "PLACE_INSIDE": ["transform.location"],
            "ALIGN_OBJECT": ["transform.location"],
            "REPLACE_MATERIAL": ["materials*"],
            "SET_MATERIAL_COLOR": ["materials*"],
            "CREATE_OBJECT": ["*"],
            "DELETE_OBJECT": ["*"],
        }
        if operation_type == "SET_CAMERA":
            suffixes[operation_type] = [
                {"location": "transform.location", "rotation_euler": "transform.rotation_euler",
                 "target": "transform.rotation_euler", "fov_degrees": "fov_degrees"}[key]
                for key in parameters if key in {"location", "rotation_euler", "target", "fov_degrees"}
            ]
        elif operation_type == "SET_LIGHT":
            suffixes[operation_type] = [
                {"location": "transform.location", "rotation_euler": "transform.rotation_euler",
                 "color": "color", "energy": "energy", "visible": "visible"}[key]
                for key in parameters if key in {"location", "rotation_euler", "color", "energy", "visible"}
            ]
        return [f"{entity_root}.{target_id}.{suffix}" for suffix in suffixes[operation_type]]

    @staticmethod
    def _postconditions(
        operation_type: str, entity_root: str, target_id: str,
        parameters: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        fields = {
            "MOVE_OBJECT": ("transform.location", "location"),
            "ROTATE_OBJECT": ("transform.rotation_euler", "rotation_euler"),
            "SCALE_OBJECT": ("transform.scale", "scale"),
        }
        if operation_type == "CREATE_OBJECT":
            return [{
                "path": f"{entity_root}.{target_id}.id", "operator": "exists",
            }]
        if operation_type == "DELETE_OBJECT":
            return [{
                "path": f"{entity_root}.{target_id}.id", "operator": "not_exists",
            }]
        if operation_type in fields:
            suffix, parameter = fields[operation_type]
            return [{
                "path": f"{entity_root}.{target_id}.{suffix}", "operator": "eq",
                "value": parameters[parameter], "tolerance": 1e-5,
            }]
        entity_fields = {
            "SET_CAMERA": {
                "location": "transform.location", "rotation_euler": "transform.rotation_euler",
                "fov_degrees": "fov_degrees",
            },
            "SET_LIGHT": {
                "location": "transform.location", "rotation_euler": "transform.rotation_euler",
                "color": "color", "energy": "energy", "visible": "visible",
            },
        }
        return [
            {"path": f"{entity_root}.{target_id}.{suffix}", "operator": "eq",
             "value": parameters[key], "tolerance": 1e-5}
            for key, suffix in entity_fields.get(operation_type, {}).items()
            if key in parameters
        ]


class AgentBlenderTransactionBackend(SceneBackend):
    """Execute VeriGraph3D actions through ABWS's transactional commit gate.

    ``execute_transaction`` receives a JSON plan and must return an ABWS result
    mapping.  The backend accepts only an explicit success/commit signal; an
    ambiguous response is rejected instead of mutating the trusted trajectory.
    """

    def __init__(
        self,
        state_reader: Callable[[], Any],
        execute_transaction: Callable[[dict[str, Any]], Any],
        *,
        mapper: AgentBlenderStateMapper | None = None,
        action_mapper: AgentBlenderActionMapper | None = None,
        plan_builder: Callable[[SemanticAction, SceneState], dict[str, Any]] | None = None,
        undo_transaction: Callable[[str | None], Any] | None = None,
        redo_transaction: Callable[[str | None], Any] | None = None,
        capability_backend: SceneBackend | None = None,
    ) -> None:
        self.state_reader = state_reader
        self.execute_transaction = execute_transaction
        self.mapper = mapper or AgentBlenderStateMapper()
        self.action_mapper = action_mapper or AgentBlenderActionMapper()
        self.plan_builder = plan_builder
        self.undo_transaction = undo_transaction
        self.redo_transaction = redo_transaction
        self.capability_backend = capability_backend
        self.last_transaction: dict[str, Any] | None = None
        self._commits: list[str | None] = []

    def read_state(self) -> SceneState:
        state = self.mapper.map(self.state_reader())
        if self.last_transaction is not None:
            state.metadata["abws_last_transaction"] = self.last_transaction
        return state

    def apply(self, action: SemanticAction) -> None:
        before = self.read_state()
        plan = (
            self.plan_builder(action, before)
            if self.plan_builder is not None
            else self.action_mapper.map(action, before)
        )
        result = _mapping(self.execute_transaction(plan))
        self.last_transaction = dict(result)
        if not self._committed(result):
            evidence = self._rejection_evidence(result, plan)
            raise AgentBlenderTransactionRejected(
                str(result.get("message", result.get("error", "ABWS rejected the transaction"))),
                evidence,
            )
        transaction_id = result.get("transaction_id", result.get("commit_id", result.get("id")))
        self._commits.append(str(transaction_id) if transaction_id is not None else None)

    def restore(self, state: SceneState) -> None:
        if self.undo_transaction is not None and self._commits:
            transaction_id = self._commits.pop()
            result = _plain(self.undo_transaction(transaction_id))
            if isinstance(result, Mapping) and not self._committed(result):
                raise AgentBlenderTransactionRejected("ABWS undo was rejected", result)
            return
        if self.capability_backend is not None:
            self.capability_backend.restore(state)
            return
        raise AgentBlenderIntegrationError(
            "Rollback requires undo_transaction or a capability_backend"
        )

    def redo(self) -> Any:
        if self.redo_transaction is None:
            raise AgentBlenderIntegrationError("ABWS redo callback is not configured")
        transaction_id = self._commits[-1] if self._commits else None
        return self.redo_transaction(transaction_id)

    def __getattr__(self, name: str) -> Any:
        if self.capability_backend is None:
            raise AttributeError(name)
        return getattr(self.capability_backend, name)

    @staticmethod
    def _committed(result: Mapping[str, Any]) -> bool:
        if result.get("committed") is True or result.get("success") is True or result.get("accepted") is True:
            return True
        return str(result.get("status", "")).lower() in {"committed", "success", "accepted"}

    @staticmethod
    def _rejection_evidence(result: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
        validation = _plain(result.get("validation", result.get("details", {})))
        validation = dict(validation) if isinstance(validation, Mapping) else {"validation": validation}
        failed = result.get("failed_constraint", validation.get("failed_constraint"))
        nested_plan = _plain(plan.get("plan", {}))
        return {
            "plan_id": nested_plan.get("plan_id") if isinstance(nested_plan, Mapping) else None,
            "base_revision": plan.get("scene_revision", plan.get("base_revision")),
            "failed_constraint": failed or "transaction_gate",
            "status": result.get("status", "rejected"),
            "property_diff": _plain(result.get("diff", result.get("property_diff", {}))),
            **validation,
        }


class AgentBlenderAtomicBlendBackend(SceneBackend):
    """Run every action on a candidate .blend and promote only validated files.

    This backend runs in the host Python process.  ABWS launches Blender for
    extraction and candidate execution, validates the actual property diff,
    then atomically writes a new authoritative revision.  The input scene is
    never opened for writing.
    """

    def __init__(
        self,
        blend_file: str | Path,
        work_directory: str | Path,
        *,
        blender_executable: str | None = None,
        mapper: AgentBlenderStateMapper | None = None,
        action_mapper: AgentBlenderActionMapper | None = None,
    ) -> None:
        try:
            from agent_blender.blender_process import BlenderProcess
            from agent_blender.models import TransactionSpec
        except ImportError as exc:
            raise AgentBlenderIntegrationError(
                "Install ABWS or add its src directory before creating the atomic backend"
            ) from exc
        source = Path(blend_file).resolve()
        if not source.is_file():
            raise AgentBlenderIntegrationError(f"Blend file not found: {source}")
        self.work_directory = Path(work_directory).resolve()
        self.work_directory.mkdir(parents=True, exist_ok=True)
        self.current_blend = source
        self.mapper = mapper or AgentBlenderStateMapper()
        self.action_mapper = action_mapper or AgentBlenderActionMapper()
        self.process = BlenderProcess(executable=blender_executable)
        self.transaction_model = TransactionSpec
        self.last_transaction: dict[str, Any] | None = None
        self._history: list[Path] = [source]
        self._states: dict[Path, SceneState] = {}

    def read_state(self) -> SceneState:
        state_path = self.work_directory / f"state_{len(self._history) - 1:04d}.json"
        world = self.process.extract(
            self.current_blend, state_path, scene_id=self.current_blend.stem
        )
        state = self.mapper.map(world)
        state.metadata["abws_blend_path"] = str(self.current_blend)
        if self.last_transaction is not None:
            state.metadata["abws_last_transaction"] = self.last_transaction
        self._states[self.current_blend] = state.clone()
        return state

    def apply(self, action: SemanticAction) -> None:
        before = self.read_state()
        payload = self.action_mapper.map(action, before)
        specification = self.transaction_model.parse_obj(payload)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", action.id)
        revision = int(before.metadata.get("abws_revision", before.timestamp)) + 1
        candidate = self.work_directory / f"revision_{revision:04d}_{safe_id}.blend"
        result = _mapping(
            self.process.execute_transaction(self.current_blend, specification, candidate)
        )
        self.last_transaction = dict(result)
        if not AgentBlenderTransactionBackend._committed(result):
            evidence = AgentBlenderTransactionBackend._rejection_evidence(result, payload)
            raise AgentBlenderTransactionRejected(
                str(result.get("error", "ABWS rejected the candidate .blend")), evidence
            )
        self.current_blend = candidate
        self._history.append(candidate)

    def restore(self, state: SceneState) -> None:
        raw_path = state.metadata.get("abws_blend_path")
        if not raw_path:
            raise AgentBlenderIntegrationError("Snapshot has no ABWS blend path")
        target = Path(str(raw_path)).resolve()
        if target not in self._history:
            raise AgentBlenderIntegrationError(
                f"Refusing rollback to an unknown blend revision: {target}"
            )
        self.current_blend = target
        self._history = self._history[: self._history.index(target) + 1]

    def export(self, output: str | Path) -> Path:
        """Copy the accepted authoritative revision to the requested result."""
        destination = Path(output).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.current_blend, destination)
        return destination


class AgentBlenderRuntimeBridge:
    """Bind a concrete ABWS runtime without coupling to one repository revision.

    ABWS is evolving, so the bridge recognises a small, explicit set of public
    method names.  It also converts the JSON plan to an annotated Pydantic model
    when the selected runtime method declares one.
    """

    _STATE_METHODS = ("read_state", "get_state", "snapshot", "extract_state")
    _STATE_ATTRIBUTES = ("state", "world_state", "current_state")
    _EXECUTE_METHODS = ("preview_and_commit", "execute_plan", "execute_transaction", "execute")
    _UNDO_METHODS = ("undo_transaction", "undo")
    _REDO_METHODS = ("redo_transaction", "redo")

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    @classmethod
    def from_factory(cls, factory_path: str) -> AgentBlenderRuntimeBridge:
        """Load ``module:callable`` and construct one runtime instance."""
        if ":" not in factory_path:
            raise AgentBlenderIntegrationError(
                "ABWS runtime factory must use the form module:callable"
            )
        module_name, attribute = factory_path.split(":", 1)
        try:
            factory = getattr(importlib.import_module(module_name), attribute)
            runtime = factory()
        except (ImportError, AttributeError, TypeError) as exc:
            raise AgentBlenderIntegrationError(
                f"Cannot create ABWS runtime from {factory_path}: {exc}"
            ) from exc
        return cls(runtime)

    def read_state(self) -> Any:
        for name in self._STATE_METHODS:
            method = getattr(self.runtime, name, None)
            if callable(method):
                return method()
        for name in self._STATE_ATTRIBUTES:
            if hasattr(self.runtime, name):
                return getattr(self.runtime, name)
        raise AgentBlenderIntegrationError(
            "ABWS runtime exposes no supported state reader: "
            + ", ".join((*self._STATE_METHODS, *self._STATE_ATTRIBUTES))
        )

    def execute(self, plan: dict[str, Any]) -> Any:
        method = self._method(self._EXECUTE_METHODS, "transaction executor")
        return method(self._coerce_plan(method, plan))

    def undo(self, transaction_id: str | None = None) -> Any:
        method = self._method(self._UNDO_METHODS, "undo method")
        return self._call_optional_id(method, transaction_id)

    def redo(self, transaction_id: str | None = None) -> Any:
        method = self._method(self._REDO_METHODS, "redo method")
        return self._call_optional_id(method, transaction_id)

    def backend(
        self,
        capability_backend: SceneBackend | None = None,
        *,
        mapper: AgentBlenderStateMapper | None = None,
        plan_builder: Callable[[SemanticAction, SceneState], dict[str, Any]] | None = None,
    ) -> AgentBlenderTransactionBackend:
        undo = self.undo if self._has_method(self._UNDO_METHODS) else None
        redo = self.redo if self._has_method(self._REDO_METHODS) else None
        return AgentBlenderTransactionBackend(
            self.read_state,
            self.execute,
            mapper=mapper,
            plan_builder=plan_builder,
            undo_transaction=undo,
            redo_transaction=redo,
            capability_backend=capability_backend,
        )

    def _method(self, names: Sequence[str], description: str) -> Callable[..., Any]:
        for name in names:
            method = getattr(self.runtime, name, None)
            if callable(method):
                return method
        raise AgentBlenderIntegrationError(
            f"ABWS runtime exposes no supported {description}: " + ", ".join(names)
        )

    def _has_method(self, names: Sequence[str]) -> bool:
        return any(callable(getattr(self.runtime, name, None)) for name in names)

    @staticmethod
    def _coerce_plan(method: Callable[..., Any], plan: dict[str, Any]) -> Any:
        try:
            parameters = list(inspect.signature(method).parameters.values())
        except (TypeError, ValueError):
            return plan
        if not parameters:
            raise AgentBlenderIntegrationError("ABWS transaction executor accepts no plan argument")
        annotation = parameters[0].annotation
        if annotation is inspect.Parameter.empty or isinstance(annotation, str):
            return plan
        validator = getattr(annotation, "model_validate", None)
        if callable(validator):
            return validator(plan)
        parser = getattr(annotation, "parse_obj", None)
        if callable(parser):
            return parser(plan)
        return plan

    @staticmethod
    def _call_optional_id(method: Callable[..., Any], transaction_id: str | None) -> Any:
        try:
            parameters = list(inspect.signature(method).parameters.values())
        except (TypeError, ValueError):
            parameters = [None]
        return method() if not parameters else method(transaction_id)
