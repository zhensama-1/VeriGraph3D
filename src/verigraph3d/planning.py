from __future__ import annotations

from .models import ActionType, Constraint, GoalSpec, Relation, SceneState, SemanticAction


class StructuredTaskParser:
    """Small reproducible Chinese/English parser baseline; VLMs can replace this adapter."""

    COLORS = {"红": (1.0, 0.0, 0.0, 1.0), "red": (1.0, 0.0, 0.0, 1.0), "蓝": (0.0, 0.0, 1.0, 1.0), "blue": (0.0, 0.0, 1.0, 1.0)}

    def parse(self, instruction: str, state: SceneState, reference_images: list[str] | None = None) -> tuple[GoalSpec, list[SemanticAction]]:
        lowered = instruction.lower()
        mentioned = sorted(
            (name for name in state.objects if name.lower() in lowered),
            key=lambda name: lowered.index(name.lower()),
        )
        goal, actions = GoalSpec(), []
        if len(mentioned) >= 2 and any(token in lowered for token in ("放在", "放到", "place", "on top")):
            target, reference = mentioned[0], mentioned[1]
            c = Constraint(target, Relation.ON_TOP_OF.value, reference)
            no_collision = Constraint(target, Relation.INTERSECTING.value, reference)
            goal.required.append(c)
            goal.forbidden.append(no_collision)
            actions.append(SemanticAction(self._id(len(actions)), ActionType.PLACE_ON, target, reference, expected_effects=[c]))
        for token, rgba in self.COLORS.items():
            if token in lowered and mentioned:
                target = mentioned[0]
                goal.visual[f"{target}.color"] = rgba
                actions.append(SemanticAction(self._id(len(actions)), ActionType.SET_MATERIAL, target, parameters={"color": rgba}))
                break
        return goal, actions

    @staticmethod
    def _id(index: int) -> str:
        return f"action_{index + 1:03d}"


class GoalPlanner:
    def plan(self, goal: GoalSpec, state: SceneState) -> list[SemanticAction]:
        actions = []
        for c in goal.required:
            if c.subject in state.cameras and c.predicate in {"location", "target", "fov_degrees"}:
                actions.append(SemanticAction(
                    f"action_{len(actions)+1:03d}", ActionType.SET_CAMERA, c.subject,
                    parameters={c.predicate: c.expected}, expected_effects=[c],
                ))
            elif c.subject in state.lights and c.predicate in {"location", "color", "energy", "visible"}:
                actions.append(SemanticAction(
                    f"action_{len(actions)+1:03d}", ActionType.SET_LIGHT, c.subject,
                    parameters={c.predicate: c.expected}, expected_effects=[c],
                ))
            elif c.predicate == Relation.ON_TOP_OF.value and c.object:
                actions.append(SemanticAction(f"action_{len(actions)+1:03d}", ActionType.PLACE_ON, c.subject, c.object, expected_effects=[c]))
            elif c.predicate in {"location", "rotation", "scale"}:
                mapping = {"location": ActionType.MOVE_OBJECT, "rotation": ActionType.ROTATE_OBJECT, "scale": ActionType.SCALE_OBJECT}
                actions.append(SemanticAction(f"action_{len(actions)+1:03d}", mapping[c.predicate], c.subject, parameters={c.predicate: c.expected}, expected_effects=[c]))
        for key, value in goal.visual.items():
            if key.endswith(".color"):
                actions.append(SemanticAction(f"action_{len(actions)+1:03d}", ActionType.SET_MATERIAL, key.rsplit(".", 1)[0], parameters={"color": value}))
        return actions
