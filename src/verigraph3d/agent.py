from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .active_view import ActiveViewSelector
from .execution import SafeExecutor
from .graph import ExecutableSceneGraph
from .metrics import RunMetrics
from .models import ActionType, GoalSpec, SceneBackend, SemanticAction
from .planning import GoalPlanner, StructuredTaskParser
from .recovery import FailureAnalyzer, LocalRepairer
from .verification import HybridVerifier, VisualVerifier


@dataclass(frozen=True, slots=True)
class AgentConfig:
    use_scene_graph: bool = True
    use_deterministic_verifier: bool = True
    use_constraint_solver: bool = True
    use_recovery: bool = True
    use_active_view: bool = False
    observation_cost: float = 0.1
    render_after_action: bool = False
    render_directory: str = "renders"
    max_repairs: int = 2


class VeriGraph3DAgent:
    def __init__(self, backend: SceneBackend, max_repairs: int = 2, config: AgentConfig | None = None, visual_verifier: VisualVerifier | None = None, task_interpreter=None) -> None:
        self.backend = backend
        self.config = config or AgentConfig(max_repairs=max_repairs)
        self.executor = SafeExecutor(backend, solve_constraints=self.config.use_constraint_solver)
        self.graph = ExecutableSceneGraph()
        self.verifier = HybridVerifier(visual=visual_verifier, deterministic_enabled=self.config.use_deterministic_verifier)
        self.failure_analyzer = FailureAnalyzer()
        self.repairer = LocalRepairer()
        self.view_selector = ActiveViewSelector()
        self.parser = task_interpreter or StructuredTaskParser()
        self.planner = GoalPlanner()
        self.max_repairs = self.config.max_repairs
        self.metrics = RunMetrics()
        self._last_render_path: str | None = None
        clients = [getattr(self.parser, "client", None), getattr(visual_verifier, "client", None)]
        self._vlm_clients = list({id(client): client for client in clients if client is not None}.values())

    def run(self, instruction: str | None = None, goal: GoalSpec | None = None, actions: list[SemanticAction] | None = None, reference_images: list[str] | None = None, uncertainties: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        self.metrics = RunMetrics()
        self.metrics.start()
        usage_before = {
            id(client): (client.usage.calls, client.usage.input_tokens, client.usage.output_tokens)
            for client in self._vlm_clients
        }
        initial = self.backend.read_state()
        if goal is None:
            if not instruction:
                raise ValueError("Either instruction or goal is required")
            goal, parsed_actions = self.parser.parse(instruction, initial, reference_images)
            actions = actions or parsed_actions
        actions = actions or self.planner.plan(goal, initial)
        allowed_changes = {action.target for action in actions}
        if self.config.use_scene_graph:
            self.graph.rebuild(initial)
            self.graph.set_goal(goal)
        trace = []
        if uncertainties:
            for uncertainty in uncertainties:
                self.graph.add_uncertainty(
                    uncertainty["subject"], uncertainty["predicate"],
                    float(uncertainty.get("confidence", 0.5)), uncertainty.get("reason", "unspecified"),
                )
        if self.config.use_active_view and uncertainties and initial.cameras:
            scorer = getattr(self.backend, "score_view", None)
            camera, gain = self.view_selector.select(initial, uncertainties, self.config.observation_cost, scorer)
            if camera is not None:
                active_name = next(iter(initial.cameras))
                observation = SemanticAction(
                    "observation_001", ActionType.SET_CAMERA, active_name,
                    parameters={"location": camera.location, "target": camera.target, "fov_degrees": camera.fov_degrees},
                )
                observed = self.executor.execute(observation)
                self.metrics.observations += observed.status == "success"
                if observed.status == "success":
                    self.metrics.observation_information_gain += gain
                    self.metrics.observation_cost += self.config.observation_cost
                    allowed_changes.add(active_name)
                trace.append({"action": observation.id, "type": "active_observation", "status": observed.status, "information_gain": gain})
        for action in actions:
            record = self.executor.execute(action)
            self.metrics.actions += 1
            self.metrics.successful_actions += record.status == "success"
            if record.status != "success":
                trace.append({"action": action.id, "status": record.status, "error": record.error})
                continue
            allowed_changes.update(record.changes)
            current = self.backend.read_state()
            self._attach_render(current, record.action.id)
            expected_changes = {action.target, *record.changes.keys()}
            report = self.verifier.verify(
                current, GoalSpec(required=action.expected_effects), reference_images,
                record.before, expected_changes,
            )
            repairs = 0
            recovery_trace = []
            while self.config.use_recovery and not report.accepted and report.decision == "local_repair" and repairs < self.max_repairs:
                failure = self.failure_analyzer.attribute(record, report)
                repair = self.repairer.propose(record.action, failure, current) if failure else None
                if repair is None:
                    break
                record = self.executor.execute(repair)
                self.metrics.actions += 1
                self.metrics.successful_actions += record.status == "success"
                self.metrics.repairs += 1
                repairs += 1
                current = self.backend.read_state()
                self._attach_render(current, record.action.id)
                expected_changes.update(record.changes)
                allowed_changes.update(record.changes)
                report = self.verifier.verify(
                    current, GoalSpec(required=action.expected_effects), reference_images,
                    record.before, expected_changes,
                )
                recovery_trace.append({
                    "failure": asdict(failure),
                    "repair_action": asdict(repair),
                    "status": record.status,
                    "accepted_after_repair": report.accepted,
                })
            if self.config.use_scene_graph:
                self.graph.rebuild(current)
                self.graph.record_action(record.action, record.status, record.changes)
            failure = self.failure_analyzer.attribute(record, report) if not report.accepted else None
            if not report.accepted and report.decision == "replan" and self.config.use_recovery and record.status == "success":
                self.executor.rollback(record.action.id)
                self.metrics.rollbacks += 1
            trace.append({
                "action": action.id, "status": record.status,
                "verification": asdict(report), "repairs": repairs,
                "recovery_trace": recovery_trace,
                "failure": asdict(failure) if failure else None,
            })
        final = self.backend.read_state()
        if self._last_render_path:
            final.metadata["render_path"] = self._last_render_path
        elif self.config.render_after_action:
            self._attach_render(final, "final")
        final_report = self.verifier.verify(final, goal, reference_images, initial, allowed_changes)
        if self.config.use_scene_graph:
            self.graph.rebuild(final)
        self.metrics.stop()
        for client in self._vlm_clients:
            calls, input_tokens, output_tokens = usage_before[id(client)]
            self.metrics.vlm_calls += client.usage.calls - calls
            self.metrics.input_tokens += client.usage.input_tokens - input_tokens
            self.metrics.output_tokens += client.usage.output_tokens - output_tokens
        return {"accepted": final_report.accepted, "decision": final_report.decision, "trace": trace, "verification": asdict(final_report), "metrics": self.metrics.to_dict(), "final_state": final.to_dict()}

    def _attach_render(self, state, label: str) -> None:
        if not self.config.render_after_action:
            return
        renderer = getattr(self.backend, "render", None)
        if not callable(renderer):
            return
        directory = Path(self.config.render_directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        path = str(directory / f"{label}.png")
        renderer(path)
        state.metadata["render_path"] = path
        self._last_render_path = path
        self.metrics.renders += 1
