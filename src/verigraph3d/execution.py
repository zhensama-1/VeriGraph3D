from __future__ import annotations

from dataclasses import asdict

from .models import ActionRecord, SceneBackend, SemanticAction
from .solver import GeometrySolver, PlanPreflight


class SafeExecutor:
    def __init__(self, backend: SceneBackend, solver: GeometrySolver | None = None, solve_constraints: bool = True) -> None:
        self.backend = backend
        self.solver = solver or GeometrySolver()
        self.preflight = PlanPreflight()
        self.solve_constraints = solve_constraints
        self.history: list[ActionRecord] = []

    def execute(self, action: SemanticAction) -> ActionRecord:
        before = self.backend.read_state()
        try:
            solved = self.solver.solve(action, before) if self.solve_constraints else action
            self.preflight.require_valid(solved, before)
            self.backend.apply(solved)
            after = self.backend.read_state()
            record = ActionRecord(solved, "success", before, after, changes=self._diff(before, after))
        except Exception as exc:
            record = ActionRecord(
                action,
                "failed",
                before,
                error=str(exc),
                error_details=dict(getattr(exc, "evidence", {}) or {}),
            )
        self.history.append(record)
        return record

    def rollback(self, action_id: str | None = None) -> ActionRecord:
        candidates = [r for r in self.history if r.status == "success"]
        if action_id:
            candidates = [r for r in candidates if r.action.id == action_id]
        if not candidates:
            raise ValueError("No successful action available for rollback")
        record = candidates[-1]
        self.backend.restore(record.before)
        record.status = "rolled_back"
        return record

    @staticmethod
    def _diff(before, after) -> dict:
        changes = {}
        for kind in ("objects", "cameras", "lights"):
            old_entities = getattr(before, kind)
            new_entities = getattr(after, kind)
            for name in set(old_entities) | set(new_entities):
                old, new = old_entities.get(name), new_entities.get(name)
                old_payload = asdict(old) if old else None
                new_payload = asdict(new) if new else None
                if kind == "objects":
                    if old_payload:
                        old_payload.pop("metadata", None)
                    if new_payload:
                        new_payload.pop("metadata", None)
                if old_payload != new_payload:
                    changes[name] = {
                        "entity_type": kind[:-1],
                        "before": old_payload,
                        "after": new_payload,
                    }
        return changes
