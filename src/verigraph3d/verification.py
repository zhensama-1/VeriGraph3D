from __future__ import annotations

from dataclasses import asdict
from typing import Protocol
import math

from .graph import ExecutableSceneGraph
from .models import CheckResult, GoalSpec, SceneState, VerificationReport
from .state import RelationCalculator


class VisualVerifier(Protocol):
    def verify(self, state: SceneState, goal: GoalSpec, reference_images: list[str]) -> tuple[float, list[str], list[str]]: ...


class RuleBasedVisualVerifier:
    """Offline baseline. Replace with a VLM adapter through the same interface."""

    def verify(self, state: SceneState, goal: GoalSpec, reference_images: list[str] | None = None):
        checks, differences = 0, []
        for key, expected in goal.visual.items():
            if "." not in key:
                continue
            name, attribute = key.split(".", 1)
            checks += 1
            obj = state.objects.get(name)
            actual = getattr(obj, attribute, None) if obj else None
            if not self._equivalent(actual, expected):
                differences.append(f"{key}: expected {expected}, got {actual}")
        score = 1.0 if checks == 0 else (checks - len(differences)) / checks
        return score, differences, ["adjust_visual_attribute"] if differences else []

    @classmethod
    def _equivalent(cls, actual, expected) -> bool:
        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            return len(actual) == len(expected) and all(cls._equivalent(a, b) for a, b in zip(actual, expected))
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return math.isclose(float(actual), float(expected), abs_tol=1e-4)
        return actual == expected


class DeterministicVerifier:
    def verify(self, state: SceneState, goal: GoalSpec, before: SceneState | None = None, expected_changes: set[str] | None = None) -> list[CheckResult]:
        graph = ExecutableSceneGraph()
        graph.rebuild(state)
        results = []
        for c in goal.required:
            results.append(CheckResult(graph.check(c), "required_constraint", f"Required: {c.subject}.{c.predicate}", {"constraint": c}))
        for c in goal.forbidden:
            results.append(CheckResult(not graph.check(c), "forbidden_constraint", f"Forbidden: {c.subject}.{c.predicate}", {"constraint": c}))
        violations = RelationCalculator().violations(state)
        for fact in violations:
            if fact.predicate == "intersecting":
                first, second = state.objects[fact.subject], state.objects[fact.object]
                depth = first.aabb.intersection_depth(second.aabb)
                results.append(CheckResult(False, "geometry_intersection", f"{fact.subject} intersects {fact.object}", {"objects": [fact.subject, fact.object], "penetration_depth": min(v for v in depth if v > 0)}))
        if before is not None and expected_changes is not None:
            changed = set()
            for kind in ("objects", "cameras", "lights"):
                old_entities, new_entities = getattr(before, kind), getattr(state, kind)
                for name in set(old_entities) | set(new_entities):
                    old, new = old_entities.get(name), new_entities.get(name)
                    old_payload, new_payload = asdict(old) if old else None, asdict(new) if new else None
                    if kind == "objects":
                        if old_payload:
                            old_payload.pop("metadata", None)
                        if new_payload:
                            new_payload.pop("metadata", None)
                    if old_payload != new_payload:
                        changed.add(name)
            unexpected = changed - expected_changes
            results.append(CheckResult(not unexpected, "unexpected_modification", "Unexpected object changes", {"objects": sorted(unexpected)}))
        return results


class HybridVerifier:
    def __init__(self, visual: VisualVerifier | None = None, visual_threshold: float = 0.8, deterministic_enabled: bool = True) -> None:
        self.visual = visual or RuleBasedVisualVerifier()
        self.deterministic = DeterministicVerifier()
        self.visual_threshold = visual_threshold
        self.deterministic_enabled = deterministic_enabled

    def verify(self, state: SceneState, goal: GoalSpec, reference_images: list[str] | None = None, before: SceneState | None = None, expected_changes: set[str] | None = None) -> VerificationReport:
        deterministic = self.deterministic.verify(state, goal, before, expected_changes) if self.deterministic_enabled else []
        score, differences, recommendations = self.visual.verify(state, goal, reference_images or [])
        hard_failures = [r for r in deterministic if not r.passed]
        if hard_failures:
            local_codes = {"required_constraint", "forbidden_constraint", "geometry_intersection"}
            affected = {obj for result in hard_failures for obj in result.evidence.get("objects", []) if obj}
            local = all(result.code in local_codes for result in hard_failures) and len(affected) <= 2
            return VerificationReport(False, "local_repair" if local else "replan", deterministic, score, differences, recommendations)
        if score < self.visual_threshold:
            return VerificationReport(False, "adjust_appearance_or_camera", deterministic, score, differences, recommendations)
        return VerificationReport(True, "accept", deterministic, score, differences, recommendations)
