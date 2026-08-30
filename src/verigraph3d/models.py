from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol
import copy
import math


Vec3 = tuple[float, float, float]


class Relation(str, Enum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    ON_TOP_OF = "on_top_of"
    INSIDE = "inside"
    INTERSECTING = "intersecting"
    IN_FRONT_OF = "in_front_of"
    VISIBLE_FROM = "visible_from"
    ATTACHED_TO = "attached_to"
    SUPPORTS = "supports"


class ActionType(str, Enum):
    CREATE_OBJECT = "create_object"
    DELETE_OBJECT = "delete_object"
    MOVE_OBJECT = "move_object"
    ROTATE_OBJECT = "rotate_object"
    SCALE_OBJECT = "scale_object"
    PLACE_ON = "place_on"
    PLACE_INSIDE = "place_inside"
    ALIGN = "align"
    SET_MATERIAL = "set_material"
    SET_CAMERA = "set_camera"
    SET_LIGHT = "set_light"


@dataclass(slots=True)
class AABB:
    minimum: Vec3
    maximum: Vec3

    @property
    def center(self) -> Vec3:
        return tuple((a + b) / 2 for a, b in zip(self.minimum, self.maximum))  # type: ignore[return-value]

    @property
    def size(self) -> Vec3:
        return tuple(b - a for a, b in zip(self.minimum, self.maximum))  # type: ignore[return-value]

    def translated(self, delta: Vec3) -> AABB:
        return AABB(_add(self.minimum, delta), _add(self.maximum, delta))

    def intersects(self, other: AABB, tolerance: float = 1e-6) -> bool:
        return all(
            self.minimum[i] < other.maximum[i] - tolerance
            and self.maximum[i] > other.minimum[i] + tolerance
            for i in range(3)
        )

    def intersection_depth(self, other: AABB) -> Vec3:
        return tuple(max(0.0, min(self.maximum[i], other.maximum[i]) - max(self.minimum[i], other.minimum[i])) for i in range(3))  # type: ignore[return-value]

    def contains(self, other: AABB, tolerance: float = 1e-6) -> bool:
        return all(self.minimum[i] - tolerance <= other.minimum[i] and self.maximum[i] + tolerance >= other.maximum[i] for i in range(3))


@dataclass(slots=True)
class SceneObject:
    name: str
    category: str = "object"
    location: Vec3 = (0.0, 0.0, 0.0)
    rotation: Vec3 = (0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)
    dimensions: Vec3 = (1.0, 1.0, 1.0)
    color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    material: str | None = None
    visible: bool = True
    parent: str | None = None
    rigid_body: bool = False
    collision_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def aabb(self) -> AABB:
        if "world_aabb" in self.metadata:
            stored = self.metadata["world_aabb"]
            origin = tuple(self.metadata.get("aabb_origin", self.location))
            delta = tuple(self.location[i] - origin[i] for i in range(3))
            return AABB(tuple(stored["minimum"]), tuple(stored["maximum"])).translated(delta)
        half = tuple(abs(self.dimensions[i] * self.scale[i]) / 2 for i in range(3))
        return AABB(tuple(self.location[i] - half[i] for i in range(3)), tuple(self.location[i] + half[i] for i in range(3)))  # type: ignore[arg-type]


@dataclass(slots=True)
class CameraState:
    name: str = "Camera"
    location: Vec3 = (6.0, -6.0, 5.0)
    target: Vec3 = (0.0, 0.0, 0.0)
    fov_degrees: float = 50.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LightState:
    name: str
    light_type: str = "POINT"
    location: Vec3 = (0.0, 0.0, 3.0)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    energy: float = 1000.0
    visible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SceneState:
    objects: dict[str, SceneObject] = field(default_factory=dict)
    cameras: dict[str, CameraState] = field(default_factory=dict)
    timestamp: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    lights: dict[str, LightState] = field(default_factory=dict)

    def clone(self) -> SceneState:
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Fact:
    subject: str
    predicate: str
    object: str | None
    value: Any = True
    confidence: float = 1.0
    source: str = "geometry"
    timestamp: int = 0


@dataclass(frozen=True, slots=True)
class Constraint:
    subject: str
    predicate: str
    object: str | None = None
    expected: Any = True
    hard: bool = True
    tolerance: float = 0.01
    weight: float = 1.0


@dataclass(slots=True)
class GoalSpec:
    required: list[Constraint] = field(default_factory=list)
    forbidden: list[Constraint] = field(default_factory=list)
    visual: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SemanticAction:
    id: str
    type: ActionType
    target: str
    reference: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Constraint] = field(default_factory=list)
    expected_effects: list[Constraint] = field(default_factory=list)
    rollback_strategy: str = "restore_snapshot"


@dataclass(slots=True)
class ActionRecord:
    action: SemanticAction
    status: str
    before: SceneState
    after: SceneState | None = None
    error: str | None = None
    error_details: dict[str, Any] = field(default_factory=dict)
    changes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CheckResult:
    passed: bool
    code: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VerificationReport:
    accepted: bool
    decision: str
    deterministic: list[CheckResult] = field(default_factory=list)
    visual_score: float | None = None
    visual_differences: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FailureAttribution:
    action_id: str
    failure_type: str
    failed_constraint: str
    objects: list[str]
    evidence: dict[str, Any]
    recommended_recovery: str


class SceneBackend(Protocol):
    def read_state(self) -> SceneState: ...
    def apply(self, action: SemanticAction) -> None: ...
    def restore(self, state: SceneState) -> None: ...


def distance(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _add(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]
