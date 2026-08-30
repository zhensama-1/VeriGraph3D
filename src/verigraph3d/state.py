from __future__ import annotations

from .models import CameraState, Fact, Relation, SceneObject, SceneState, distance


class RelationCalculator:
    """Deterministic spatial predicates based on world-space AABBs."""

    def __init__(self, tolerance: float = 0.02, front_axis: int = 1, front_sign: int = -1) -> None:
        self.tolerance = tolerance
        if front_axis not in {0, 1, 2} or front_sign not in {-1, 1}:
            raise ValueError("front_axis must be 0..2 and front_sign must be -1 or 1")
        self.front_axis = front_axis
        self.front_sign = front_sign

    def facts(self, state: SceneState) -> list[Fact]:
        facts: list[Fact] = []
        objects = list(state.objects.values())
        for i, a in enumerate(objects):
            for b in objects[i + 1 :]:
                facts.extend(self._pair(a, b, state.timestamp))
        for obj in objects:
            for camera in state.cameras.values():
                score = obj.metadata.get("camera_visibility", {}).get(camera.name)
                confidence = float(score) if score is not None else 1.0
                facts.append(Fact(obj.name, Relation.VISIBLE_FROM.value, camera.name, self.visible(obj, camera), confidence, "engine_ray_cast" if score is not None else "geometry", state.timestamp))
        return facts

    def _pair(self, a: SceneObject, b: SceneObject, timestamp: int) -> list[Fact]:
        out: list[Fact] = []
        out.extend(self._directed(a, b, timestamp))
        out.extend(self._directed(b, a, timestamp))
        return out

    def _directed(self, a: SceneObject, b: SceneObject, timestamp: int) -> list[Fact]:
        t = self.tolerance
        predicates = {
            Relation.LEFT_OF: a.aabb.maximum[0] <= b.aabb.minimum[0] + t,
            Relation.RIGHT_OF: a.aabb.minimum[0] >= b.aabb.maximum[0] - t,
            Relation.BELOW: a.aabb.maximum[2] <= b.aabb.minimum[2] + t,
            Relation.ABOVE: a.aabb.minimum[2] >= b.aabb.maximum[2] - t,
            Relation.INTERSECTING: a.aabb.intersects(b.aabb, t),
            Relation.INSIDE: b.aabb.contains(a.aabb, t),
            Relation.ON_TOP_OF: abs(a.aabb.minimum[2] - b.aabb.maximum[2]) <= t and self._xy_overlap(a, b),
            Relation.SUPPORTS: abs(b.aabb.minimum[2] - a.aabb.maximum[2]) <= t and self._xy_overlap(a, b),
            Relation.IN_FRONT_OF: self._in_front(a, b),
            Relation.ATTACHED_TO: a.parent == b.name,
        }
        return [Fact(a.name, rel.value, b.name, value, 1.0, "geometry", timestamp) for rel, value in predicates.items()]

    def violations(self, state: SceneState) -> list[Fact]:
        violations = []
        seen_intersections: set[tuple[str, str]] = set()
        for fact in self.facts(state):
            if fact.predicate != Relation.INTERSECTING.value or not fact.value or fact.object is None:
                continue
            pair = tuple(sorted((fact.subject, fact.object)))
            if pair not in seen_intersections:
                violations.append(fact)
                seen_intersections.add(pair)
        for obj in state.objects.values():
            if obj.metadata.get("allow_floating") or obj.aabb.minimum[2] <= self.tolerance:
                continue
            supported = any(
                abs(obj.aabb.minimum[2] - other.aabb.maximum[2]) <= self.tolerance and self._xy_overlap(obj, other)
                for other in state.objects.values() if other.name != obj.name
            )
            if not supported:
                violations.append(Fact(obj.name, "floating", None, True, 1.0, "geometry", state.timestamp))
        return violations

    def visible(self, obj: SceneObject, camera: CameraState) -> bool:
        if not obj.visible:
            return False
        engine_score = obj.metadata.get("camera_visibility", {}).get(camera.name)
        if engine_score is not None:
            return float(engine_score) >= 0.1
        to_obj = tuple(obj.location[i] - camera.location[i] for i in range(3))
        to_target = tuple(camera.target[i] - camera.location[i] for i in range(3))
        norm = distance((0, 0, 0), to_obj) * distance((0, 0, 0), to_target)
        if norm == 0:
            return True
        cosine = sum(a * b for a, b in zip(to_obj, to_target)) / norm
        import math
        return cosine >= math.cos(math.radians(camera.fov_degrees / 2))

    @staticmethod
    def _xy_overlap(a: SceneObject, b: SceneObject) -> bool:
        return all(a.aabb.minimum[i] <= b.aabb.maximum[i] and a.aabb.maximum[i] >= b.aabb.minimum[i] for i in (0, 1))

    def _in_front(self, a: SceneObject, b: SceneObject) -> bool:
        axis = self.front_axis
        if self.front_sign < 0:
            return a.aabb.maximum[axis] <= b.aabb.minimum[axis] + self.tolerance
        return a.aabb.minimum[axis] >= b.aabb.maximum[axis] - self.tolerance


class StateReader:
    def __init__(self, backend) -> None:
        self.backend = backend

    def read(self) -> SceneState:
        state = self.backend.read_state()
        state.metadata["violations"] = [f.__dict__ if hasattr(f, "__dict__") else {"subject": f.subject, "predicate": f.predicate, "object": f.object, "value": f.value} for f in RelationCalculator().violations(state)]
        return state
