from __future__ import annotations

from dataclasses import replace
from typing import Any

from .models import ActionType, CheckResult, Relation, SceneObject, SceneState, SemanticAction
from .state import RelationCalculator


class GeometrySolver:
    """Converts semantic actions into deterministic parameters."""

    def __init__(self, clearance: float = 0.01, search_step: float = 0.1, search_radius: float = 1.0) -> None:
        self.clearance = clearance
        self.search_step = search_step
        self.search_radius = search_radius

    def solve(self, action: SemanticAction, state: SceneState) -> SemanticAction:
        if action.type == ActionType.PLACE_ON:
            return self._place_on(action, state)
        if action.type == ActionType.PLACE_INSIDE:
            return self._place_inside(action, state)
        if action.type == ActionType.ALIGN:
            return self._align(action, state)
        return action

    def _place_on(self, action: SemanticAction, state: SceneState) -> SemanticAction:
        target, reference = self._objects(action, state)
        z = reference.aabb.maximum[2] + abs(target.dimensions[2] * target.scale[2]) / 2 + self.clearance
        center = reference.aabb.center
        candidates = [(center[0], center[1], z)]
        rings = int(self.search_radius / self.search_step)
        for ring in range(1, rings + 1):
            d = ring * self.search_step
            candidates.extend([(center[0] + d, center[1], z), (center[0] - d, center[1], z), (center[0], center[1] + d, z), (center[0], center[1] - d, z)])
        location = next((p for p in candidates if self._collision_free(target, p, state, {target.name, reference.name})), None)
        if location is None:
            raise ValueError(f"No collision-free placement found for {target.name} on {reference.name}")
        params = {**action.parameters, "location": location, "clearance": self.clearance}
        return replace(action, parameters=params)

    def _place_inside(self, action: SemanticAction, state: SceneState) -> SemanticAction:
        target, container = self._objects(action, state)
        if any(target.aabb.size[i] + 2 * self.clearance > container.aabb.size[i] for i in range(3)):
            raise ValueError(f"{target.name} does not fit inside {container.name}")
        return replace(action, parameters={**action.parameters, "location": container.aabb.center})

    def _align(self, action: SemanticAction, state: SceneState) -> SemanticAction:
        target, reference = self._objects(action, state)
        axes = action.parameters.get("axes", "xy")
        location = list(target.location)
        for axis, index in zip("xyz", range(3)):
            if axis in axes:
                location[index] = reference.location[index]
        return replace(action, parameters={**action.parameters, "location": tuple(location)})

    def _collision_free(self, target: SceneObject, location, state: SceneState, ignored: set[str]) -> bool:
        moved = replace(target, location=tuple(location))
        return not any(moved.aabb.intersects(other.aabb) for other in state.objects.values() if other.name not in ignored and other.collision_enabled)

    @staticmethod
    def _objects(action: SemanticAction, state: SceneState) -> tuple[SceneObject, SceneObject]:
        if action.target not in state.objects:
            raise ValueError(f"Target object not found: {action.target}")
        if not action.reference or action.reference not in state.objects:
            raise ValueError(f"Reference object not found: {action.reference}")
        return state.objects[action.target], state.objects[action.reference]


class PlanPreflight:
    def check(self, action: SemanticAction, state: SceneState) -> list[CheckResult]:
        results: list[CheckResult] = []
        exists = action.target in state.objects
        if action.type == ActionType.SET_CAMERA:
            exists = action.target in state.cameras
        if action.type == ActionType.SET_LIGHT:
            exists = action.target in state.lights
        if action.type != ActionType.CREATE_OBJECT:
            results.append(CheckResult(exists, "target_exists", f"Target {action.target} {'exists' if exists else 'is missing'}"))
        else:
            results.append(CheckResult(
                not exists, "target_is_new",
                f"Target {action.target} {'is new' if not exists else 'already exists'}",
            ))
            primitive = str(action.parameters.get("primitive", "cube")).lower()
            results.append(CheckResult(
                primitive in {"cube", "sphere", "cylinder", "cone", "plane"},
                "allowed_primitive", f"Primitive {primitive} must be in the safe whitelist",
            ))
            dimensions = action.parameters.get("dimensions", (1.0, 1.0, 1.0))
            valid_dimensions = (
                isinstance(dimensions, (list, tuple)) and len(dimensions) == 3
                and all(isinstance(value, (int, float)) and value > 0 for value in dimensions)
            )
            results.append(CheckResult(
                valid_dimensions, "valid_dimensions",
                "Created object dimensions must contain three positive numbers",
            ))
            location = action.parameters.get(
                "location", (0.0, 0.0, float(dimensions[2]) / 2)
            ) if valid_dimensions else None
            if (
                valid_dimensions and isinstance(location, (list, tuple))
                and len(location) == 3
                and all(isinstance(value, (int, float)) for value in location)
            ):
                candidate = SceneObject(
                    action.target, location=tuple(location),
                    dimensions=tuple(dimensions),
                    scale=tuple(action.parameters.get("scale", (1.0, 1.0, 1.0))),
                )
                collisions = [
                    obj.name for obj in state.objects.values()
                    if obj.collision_enabled and candidate.aabb.intersects(obj.aabb)
                ]
                results.append(CheckResult(
                    not collisions, "collision_prediction", "Predicted collisions",
                    {"objects": collisions},
                ))
        if action.reference:
            ref_exists = action.reference in state.objects
            results.append(CheckResult(ref_exists, "reference_exists", f"Reference {action.reference} {'exists' if ref_exists else 'is missing'}"))
        if "location" in action.parameters:
            loc = action.parameters["location"]
            legal = isinstance(loc, (list, tuple)) and len(loc) == 3 and all(isinstance(v, (int, float)) for v in loc)
            results.append(CheckResult(legal, "valid_location", "Location must contain three finite numbers"))
        for precondition in action.preconditions:
            from .graph import ExecutableSceneGraph
            graph = ExecutableSceneGraph()
            graph.rebuild(state)
            ok = graph.check(precondition)
            results.append(CheckResult(ok, "precondition", f"Precondition {precondition.predicate}", {"constraint": str(precondition)}))
        if action.target in state.objects and "location" in action.parameters:
            moved = replace(state.objects[action.target], location=tuple(action.parameters["location"]))
            collisions = [o.name for o in state.objects.values() if o.name != moved.name and moved.aabb.intersects(o.aabb) and o.name != action.reference]
            results.append(CheckResult(not collisions, "collision_prediction", "Predicted collisions", {"objects": collisions}))
        return results

    def require_valid(self, action: SemanticAction, state: SceneState) -> None:
        failed = [r for r in self.check(action, state) if not r.passed]
        if failed:
            raise ValueError("Preflight rejected action: " + "; ".join(r.code for r in failed))
