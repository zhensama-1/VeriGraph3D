from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import Constraint, Fact, GoalSpec, SceneState, SemanticAction
from .state import RelationCalculator


class ExecutableSceneGraph:
    """In-memory executable dynamic scene graph with provenance and history."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.facts: dict[tuple[str, str, str | None], Fact] = {}
        self.goal = GoalSpec()
        self.actions: list[dict[str, Any]] = []
        self.uncertainties: list[dict[str, Any]] = []

    def rebuild(self, state: SceneState) -> None:
        self.nodes = {
            obj.name: {"type": "container" if obj.category in {"container", "region"} else "object", **asdict(obj)}
            for obj in state.objects.values()
        }
        self.nodes.update({cam.name: {"type": "camera", **asdict(cam)} for cam in state.cameras.values()})
        self.nodes.update({light.name: {"type": "light", **asdict(light)} for light in state.lights.values()})
        self.facts = {(f.subject, f.predicate, f.object): f for f in RelationCalculator().facts(state)}
        for obj in state.objects.values():
            existence = Fact(obj.name, "exists", None, True, 1.0, "engine", state.timestamp)
            self.facts[(obj.name, "exists", None)] = existence
            for attr in ("location", "rotation", "scale", "color", "material", "visible"):
                fact = Fact(obj.name, attr, None, getattr(obj, attr), 1.0, "engine", state.timestamp)
                self.facts[(obj.name, attr, None)] = fact
            if obj.material:
                material_id = f"material:{obj.material}"
                self.nodes.setdefault(material_id, {"type": "material", "name": obj.material})
                fact = Fact(obj.name, "uses_material", material_id, True, 1.0, "engine", state.timestamp)
                self.facts[(fact.subject, fact.predicate, fact.object)] = fact
        for camera in state.cameras.values():
            for attr in ("location", "target", "fov_degrees"):
                fact = Fact(camera.name, attr, None, getattr(camera, attr), 1.0, "engine", state.timestamp)
                self.facts[(camera.name, attr, None)] = fact
        for light in state.lights.values():
            for attr in ("location", "color", "energy", "visible", "light_type"):
                fact = Fact(light.name, attr, None, getattr(light, attr), 1.0, "engine", state.timestamp)
                self.facts[(light.name, attr, None)] = fact
        self._sync_control_graph(state.timestamp)

    def set_goal(self, goal: GoalSpec) -> None:
        self.goal = goal
        self._sync_control_graph()

    def query(self, subject: str, predicate: str, obj: str | None = None) -> Fact | None:
        return self.facts.get((subject, predicate, obj))

    def check(self, constraint: Constraint) -> bool:
        fact = self.query(constraint.subject, constraint.predicate, constraint.object)
        if fact is None:
            if constraint.predicate == "exists":
                return constraint.expected is False
            return False
        if isinstance(fact.value, (tuple, list)) and isinstance(constraint.expected, (tuple, list)):
            return all(abs(a - b) <= constraint.tolerance for a, b in zip(fact.value, constraint.expected))
        if (
            isinstance(fact.value, (int, float))
            and not isinstance(fact.value, bool)
            and isinstance(constraint.expected, (int, float))
            and not isinstance(constraint.expected, bool)
        ):
            return abs(float(fact.value) - float(constraint.expected)) <= constraint.tolerance
        return fact.value == constraint.expected

    def goal_status(self) -> dict[str, list[Constraint]]:
        required_failed = [c for c in self.goal.required if not self.check(c)]
        forbidden_failed = [c for c in self.goal.forbidden if self.check(c)]
        return {"required_failed": required_failed, "forbidden_failed": forbidden_failed}

    def record_action(self, action: SemanticAction, status: str, changes: dict[str, Any] | None = None) -> None:
        self.actions.append({"action": asdict(action), "status": status, "changes": changes or {}})
        self._sync_control_graph()

    def add_uncertainty(self, subject: str, predicate: str, confidence: float, reason: str) -> None:
        self.uncertainties.append({"subject": subject, "predicate": predicate, "confidence": confidence, "reason": reason})

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "facts": [asdict(f) for f in self.facts.values()],
            "goal": asdict(self.goal),
            "actions": self.actions,
            "uncertainties": self.uncertainties,
        }

    def diff(self, previous: ExecutableSceneGraph) -> dict[str, Any]:
        current_nodes, previous_nodes = set(self.nodes), set(previous.nodes)
        current_facts, previous_facts = set(self.facts), set(previous.facts)
        changed_nodes = sorted(name for name in current_nodes & previous_nodes if self.nodes[name] != previous.nodes[name])
        changed_facts = sorted(
            (key for key in current_facts & previous_facts if self.facts[key] != previous.facts[key]),
            key=str,
        )
        return {
            "added_nodes": sorted(current_nodes - previous_nodes),
            "removed_nodes": sorted(previous_nodes - current_nodes),
            "changed_nodes": changed_nodes,
            "added_facts": [asdict(self.facts[key]) for key in sorted(current_facts - previous_facts, key=str)],
            "removed_facts": [asdict(previous.facts[key]) for key in sorted(previous_facts - current_facts, key=str)],
            "changed_facts": [
                {"before": asdict(previous.facts[key]), "after": asdict(self.facts[key])}
                for key in changed_facts
            ],
        }

    def _sync_control_graph(self, timestamp: int = 0) -> None:
        for name in [name for name in self.nodes if name.startswith(("constraint:", "action:"))]:
            self.nodes.pop(name, None)
        for key in [key for key in self.facts if key[0].startswith(("constraint:", "action:"))]:
            self.facts.pop(key, None)
        groups = (("required", self.goal.required), ("forbidden", self.goal.forbidden))
        for group, constraints in groups:
            for index, constraint in enumerate(constraints, 1):
                node_id = f"constraint:{group}:{index:03d}"
                self.nodes[node_id] = {"type": "constraint", "group": group, **asdict(constraint)}
                link = Fact(node_id, "constrains", constraint.subject, True, 1.0, "goal", timestamp)
                self.facts[(link.subject, link.predicate, link.object)] = link
                if constraint.object:
                    reference = Fact(node_id, "references", constraint.object, True, 1.0, "goal", timestamp)
                    self.facts[(reference.subject, reference.predicate, reference.object)] = reference
        for record in self.actions:
            action = record["action"]
            node_id = f"action:{action['id']}"
            action_type = action["type"].value if hasattr(action["type"], "value") else str(action["type"])
            self.nodes[node_id] = {"type": "action", "action_type": action_type, "status": record["status"], "changes": record["changes"]}
            target = Fact(node_id, "acts_on", action["target"], True, 1.0, "execution", timestamp)
            self.facts[(target.subject, target.predicate, target.object)] = target
            if action.get("reference"):
                reference = Fact(node_id, "uses_reference", action["reference"], True, 1.0, "execution", timestamp)
                self.facts[(reference.subject, reference.predicate, reference.object)] = reference
