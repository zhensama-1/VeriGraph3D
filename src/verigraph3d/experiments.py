from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from math import comb
from pathlib import Path
import random
from statistics import mean
from typing import Any

from .agent import AgentConfig, VeriGraph3DAgent
from .backends import FaultInjectingBackend, FaultRule, MemoryBackend
from .dataset import TaskDataset
from .verification import HybridVerifier


@dataclass(frozen=True, slots=True)
class ExperimentVariant:
    name: str
    scene_graph: bool = True
    deterministic_verifier: bool = True
    constraint_solver: bool = True
    failure_recovery: bool = True
    active_view: bool = False

    def agent_config(self) -> AgentConfig:
        return AgentConfig(
            use_scene_graph=self.scene_graph,
            use_deterministic_verifier=self.deterministic_verifier,
            use_constraint_solver=self.constraint_solver,
            use_recovery=self.failure_recovery,
            use_active_view=self.active_view,
        )


ABLATIONS = [
    ExperimentVariant("pure_visual", False, False, False, False),
    ExperimentVariant("scene_memory", False, False, False, False),
    ExperimentVariant("dynamic_scene_graph", True, False, False, False),
    ExperimentVariant("deterministic_verifier", True, True, False, False),
    ExperimentVariant("constraint_solver", True, True, True, False),
    ExperimentVariant("verigraph3d_full", True, True, True, True),
]

ACTIVE_VIEW_EXPERIMENTS = [
    ExperimentVariant("verigraph3d", True, True, True, True, False),
    ExperimentVariant("verigraph3d_active", True, True, True, True, True),
]


class ExperimentRunner:
    def __init__(self, dataset: TaskDataset) -> None:
        self.dataset = dataset

    def run(self, variants: list[ExperimentVariant] | None = None) -> dict[str, Any]:
        variants = variants or ABLATIONS
        runs = []
        for variant in variants:
            for task in self.dataset.tasks:
                random.seed(task.seed)
                memory = MemoryBackend(self.dataset.initial_state(task))
                fault_specs = list(task.metadata.get("faults", []))
                backend = (
                    FaultInjectingBackend(memory, [FaultRule(**spec) for spec in fault_specs])
                    if fault_specs else memory
                )
                agent = VeriGraph3DAgent(backend, config=variant.agent_config())
                result = agent.run(
                    goal=task.goal, actions=task.actions or None,
                    reference_images=task.reference_images,
                    uncertainties=task.uncertainties,
                )
                oracle = HybridVerifier(deterministic_enabled=True).verify(
                    backend.read_state(), task.goal, task.reference_images
                )
                constraint_results = [
                    check for check in oracle.deterministic
                    if check.code in {"required_constraint", "forbidden_constraint"}
                ]
                recovery_events = [
                    event for item in result["trace"]
                    for event in item.get("recovery_trace", [])
                ]
                failed_actions = sum(
                    item.get("status") == "failed" for item in result["trace"]
                )
                expected_failure = task.metadata.get("expected_failure_type")
                attributed = (
                    recovery_events[0]["failure"]["failure_type"]
                    if recovery_events else None
                )
                runs.append({
                    "variant": variant.name,
                    "task_id": task.id,
                    "task_category": task.metadata.get("category", "unspecified"),
                    "difficulty": task.metadata.get("difficulty", "unspecified"),
                    "seed": task.seed,
                    "reported_accepted": result["accepted"],
                    "ground_truth_success": oracle.accepted,
                    "false_acceptance": result["accepted"] and not oracle.accepted,
                    "constraint_satisfaction_rate": (
                        mean(float(check.passed) for check in constraint_results)
                        if constraint_results else 1.0
                    ),
                    "severe_physical_violation": any(
                        not check.passed and check.code == "geometry_intersection"
                        for check in oracle.deterministic
                    ),
                    "invalid_actions": failed_actions,
                    "repair_attempted": bool(recovery_events),
                    "local_repair_success": bool(recovery_events) and oracle.accepted,
                    "expected_failure_type": expected_failure,
                    "attributed_failure_type": attributed,
                    "failure_attribution_correct": (
                        attributed == expected_failure if expected_failure else None
                    ),
                    "oracle_verification": asdict(oracle),
                    **result,
                })
        return {
            "schema_version": "1.1",
            "runs": runs,
            "summary": self.summarize(runs),
            "category_summary": self.summarize_by_category(runs),
            "comparisons": self.compare_with_full(runs),
        }

    @staticmethod
    def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
        summary = {}
        for variant in sorted({run["variant"] for run in runs}):
            subset = [run for run in runs if run["variant"] == variant]
            action_count = sum(r["metrics"]["actions"] for r in subset)
            observation_count = sum(r["metrics"]["observations"] for r in subset)
            information_gain = sum(
                r["metrics"]["observation_information_gain"] for r in subset
            )
            observation_cost = sum(r["metrics"]["observation_cost"] for r in subset)
            faulted = [r for r in subset if r["expected_failure_type"] is not None]
            repaired = [r for r in subset if r["repair_attempted"]]
            successes = sum(bool(r["ground_truth_success"]) for r in subset)
            false_acceptances = sum(bool(r["false_acceptance"]) for r in subset)
            summary[variant] = {
                "tasks": len(subset),
                "task_success_rate": mean(float(r["ground_truth_success"]) for r in subset),
                "task_success_ci95": ExperimentRunner._wilson(successes, len(subset)),
                "agent_acceptance_rate": mean(float(r["reported_accepted"]) for r in subset),
                "false_acceptance_rate": mean(float(r["false_acceptance"]) for r in subset),
                "false_acceptance_ci95": ExperimentRunner._wilson(false_acceptances, len(subset)),
                "constraint_satisfaction_rate": mean(r["constraint_satisfaction_rate"] for r in subset),
                "severe_physical_violation_rate": mean(float(r["severe_physical_violation"]) for r in subset),
                "invalid_action_rate": (
                    sum(r["invalid_actions"] for r in subset) / action_count
                    if action_count else 0.0
                ),
                "mean_actions": mean(r["metrics"]["actions"] for r in subset),
                "mean_repairs": mean(r["metrics"]["repairs"] for r in subset),
                "mean_elapsed_seconds": mean(r["metrics"]["elapsed_seconds"] for r in subset),
                "mean_rollbacks": mean(r["metrics"]["rollbacks"] for r in subset),
                "mean_observations": mean(r["metrics"]["observations"] for r in subset),
                "mean_information_gain_per_observation": (
                    information_gain / observation_count if observation_count else None
                ),
                "mean_observation_cost": mean(
                    r["metrics"]["observation_cost"] for r in subset
                ),
                "information_gain_cost_ratio": (
                    information_gain / observation_cost if observation_cost else None
                ),
                "failure_attribution_accuracy": (
                    mean(float(r["failure_attribution_correct"]) for r in faulted)
                    if faulted else None
                ),
                "local_repair_success_rate": (
                    mean(float(r["local_repair_success"]) for r in repaired)
                    if repaired else None
                ),
            }
        return summary

    @staticmethod
    def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
        if total <= 0:
            return [0.0, 0.0]
        proportion = successes / total
        denominator = 1 + z * z / total
        center = (proportion + z * z / (2 * total)) / denominator
        margin = z * (
            (proportion * (1 - proportion) / total + z * z / (4 * total * total)) ** 0.5
        ) / denominator
        return [max(0.0, center - margin), min(1.0, center + margin)]

    @staticmethod
    def compare_with_full(runs: list[dict[str, Any]]) -> dict[str, Any]:
        """Paired task comparison against the full model using exact McNemar."""
        variants = {run["variant"] for run in runs}
        reference = "verigraph3d_full" if "verigraph3d_full" in variants else None
        if reference is None:
            return {}
        full = {
            run["task_id"]: bool(run["ground_truth_success"])
            for run in runs if run["variant"] == reference
        }
        comparisons = {}
        for variant in sorted(variants - {reference}):
            baseline = {
                run["task_id"]: bool(run["ground_truth_success"])
                for run in runs if run["variant"] == variant
            }
            common = sorted(set(full) & set(baseline))
            baseline_only = sum(baseline[key] and not full[key] for key in common)
            full_only = sum(full[key] and not baseline[key] for key in common)
            discordant = baseline_only + full_only
            if discordant:
                tail = sum(
                    comb(discordant, index) for index in range(min(baseline_only, full_only) + 1)
                ) / (2 ** discordant)
                p_value = min(1.0, 2 * tail)
            else:
                p_value = 1.0
            comparisons[variant] = {
                "reference": reference,
                "paired_tasks": len(common),
                "success_rate_delta": (
                    mean(float(full[key]) for key in common)
                    - mean(float(baseline[key]) for key in common)
                    if common else 0.0
                ),
                "baseline_only_successes": baseline_only,
                "full_only_successes": full_only,
                "mcnemar_exact_p": p_value,
            }
        return comparisons

    @staticmethod
    def summarize_by_category(runs: list[dict[str, Any]]) -> dict[str, Any]:
        output = {}
        for variant in sorted({run["variant"] for run in runs}):
            output[variant] = {}
            variant_runs = [run for run in runs if run["variant"] == variant]
            for category in sorted({run["task_category"] for run in variant_runs}):
                subset = [run for run in variant_runs if run["task_category"] == category]
                successes = sum(bool(run["ground_truth_success"]) for run in subset)
                output[variant][category] = {
                    "tasks": len(subset),
                    "task_success_rate": successes / len(subset),
                    "task_success_ci95": ExperimentRunner._wilson(successes, len(subset)),
                    "agent_acceptance_rate": mean(
                        float(run["reported_accepted"]) for run in subset
                    ),
                    "false_acceptance_rate": mean(
                        float(run["false_acceptance"]) for run in subset
                    ),
                    "constraint_satisfaction_rate": mean(
                        run["constraint_satisfaction_rate"] for run in subset
                    ),
                    "mean_actions": mean(run["metrics"]["actions"] for run in subset),
                }
        return output

    @staticmethod
    def save(result: dict[str, Any], path: str | Path) -> None:
        Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def save_summary_csv(result: dict[str, Any], path: str | Path) -> None:
        rows = [{"variant": variant, **metrics} for variant, metrics in result["summary"].items()]
        scalar_keys = [
            key for key in rows[0]
            if not any(isinstance(row.get(key), (list, dict)) for row in rows)
        ] if rows else ["variant"]
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def save_summary_markdown(result: dict[str, Any], path: str | Path) -> None:
        headers = [
            "Method", "Success", "Constraint", "False accept", "Invalid action",
            "Physical violation", "Attribution", "Repair success", "Mean actions",
        ]
        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
        ]
        percentage = lambda value: "—" if value is None else f"{100 * value:.1f}%"
        for variant, metrics in result["summary"].items():
            lines.append("| " + " | ".join([
                variant,
                percentage(metrics["task_success_rate"]),
                percentage(metrics["constraint_satisfaction_rate"]),
                percentage(metrics["false_acceptance_rate"]),
                percentage(metrics["invalid_action_rate"]),
                percentage(metrics["severe_physical_violation_rate"]),
                percentage(metrics["failure_attribution_accuracy"]),
                percentage(metrics["local_repair_success_rate"]),
                f"{metrics['mean_actions']:.2f}",
            ]) + " |")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def save_category_csv(result: dict[str, Any], path: str | Path) -> None:
        rows = [
            {"variant": variant, "category": category, **metrics}
            for variant, categories in result["category_summary"].items()
            for category, metrics in categories.items()
        ]
        scalar_keys = [
            key for key in rows[0]
            if not any(isinstance(row.get(key), (list, dict)) for row in rows)
        ] if rows else ["variant", "category"]
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
