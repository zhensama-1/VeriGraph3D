from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import mean
import sys
import tempfile
import time
from typing import Any

from .agent import VeriGraph3DAgent
from .dataset import TaskCase, TaskDataset


def select_tasks(
    dataset: TaskDataset, task_ids: list[str], categories: list[str], limit: int | None,
) -> list[TaskCase]:
    selected = [
        task for task in dataset.tasks
        if (not task_ids or task.id in task_ids)
        and (not categories or task.metadata.get("category") in categories)
    ]
    missing = sorted(set(task_ids) - {task.id for task in selected})
    if missing:
        raise ValueError(f"Unknown or filtered task IDs: {', '.join(missing)}")
    return selected[:limit] if limit is not None else selected


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [run for run in runs if run.get("accepted")]
    category_summary = {}
    for category in sorted({run["category"] for run in runs}):
        subset = [run for run in runs if run["category"] == category]
        category_summary[category] = {
            "tasks": len(subset),
            "accepted": sum(bool(run.get("accepted")) for run in subset),
            "success_rate": mean(float(bool(run.get("accepted"))) for run in subset),
            "mean_seconds": mean(float(run.get("elapsed_seconds", 0)) for run in subset),
        }
    return {
        "tasks": len(runs),
        "accepted": len(successful),
        "task_success_rate": len(successful) / len(runs) if runs else 0.0,
        "mean_seconds": mean(float(run.get("elapsed_seconds", 0)) for run in runs) if runs else 0.0,
        "total_blender_actions": sum(int(run.get("actions", 0)) for run in runs),
        "failed_task_ids": [run["task_id"] for run in runs if not run.get("accepted")],
        "categories": category_summary,
    }


def write_csv(runs: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "task_id", "category", "accepted", "decision", "actions",
        "successful_actions", "repairs", "rollbacks", "elapsed_seconds",
        "final_revision", "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: run.get(key) for key in fields} for run in runs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a stratified VeriGraph3D task subset through real ABWS Blender transactions"
    )
    parser.add_argument("blend_file", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--abws-src", type=Path, default=Path("external/AgentBlender-World-State/src"))
    parser.add_argument("--blender-executable", default=r"E:\Blender\blender.exe")
    args = parser.parse_args(argv)

    if not args.blend_file.is_file():
        raise FileNotFoundError(args.blend_file)
    sys.path.insert(0, str(args.abws_src.resolve()))
    from .agentblender import AgentBlenderAtomicBlendBackend

    dataset = TaskDataset.load(args.dataset)
    tasks = select_tasks(dataset, args.task_id, args.category, args.limit)
    if not tasks:
        raise ValueError("No tasks selected")
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(args.blend_file.read_bytes()).hexdigest()
    runs = []
    started = time.perf_counter()
    for task in tasks:
        task_started = time.perf_counter()
        task_directory = output / task.id
        task_directory.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix=f"vg3d_{task.id}_") as work:
                backend = AgentBlenderAtomicBlendBackend(
                    args.blend_file, work,
                    blender_executable=args.blender_executable,
                )
                result = VeriGraph3DAgent(backend).run(
                    instruction=task.instruction, goal=task.goal,
                    actions=task.actions or None,
                    reference_images=task.reference_images,
                    uncertainties=task.uncertainties,
                )
                if result["accepted"]:
                    backend.export(task_directory / "result.blend")
            (task_directory / "report.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            metrics = result["metrics"]
            final_revision = result["final_state"]["metadata"].get("abws_revision")
            run = {
                "task_id": task.id,
                "category": task.metadata.get("category", "unspecified"),
                "accepted": result["accepted"], "decision": result["decision"],
                "actions": metrics["actions"],
                "successful_actions": metrics["successful_actions"],
                "repairs": metrics["repairs"], "rollbacks": metrics["rollbacks"],
                "elapsed_seconds": time.perf_counter() - task_started,
                "final_revision": final_revision, "error": None,
                "report": str(task_directory / "report.json"),
                "blend": str(task_directory / "result.blend") if result["accepted"] else None,
            }
        except Exception as exc:
            run = {
                "task_id": task.id,
                "category": task.metadata.get("category", "unspecified"),
                "accepted": False, "decision": "execution_error", "actions": 0,
                "successful_actions": 0, "repairs": 0, "rollbacks": 0,
                "elapsed_seconds": time.perf_counter() - task_started,
                "final_revision": None, "error": str(exc), "report": None, "blend": None,
            }
        runs.append(run)
        print(json.dumps({
            "task_id": run["task_id"], "accepted": run["accepted"],
            "elapsed_seconds": round(run["elapsed_seconds"], 3),
        }, ensure_ascii=False), flush=True)

    report = {
        "schema_version": "1.0",
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "blender_executable": str(Path(args.blender_executable).resolve()),
            "source_blend": str(args.blend_file.resolve()), "source_sha256": source_hash,
            "dataset": str(args.dataset.resolve()),
        },
        "source_integrity": {
            "sha256_before": source_hash,
            "sha256_after": sha256(args.blend_file.read_bytes()).hexdigest(),
        },
        "wall_seconds": time.perf_counter() - started,
        "runs": runs, "summary": summarize(runs),
    }
    report["source_integrity"]["unchanged"] = (
        report["source_integrity"]["sha256_before"]
        == report["source_integrity"]["sha256_after"]
    )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(runs, output / "summary.csv")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["accepted"] == report["summary"]["tasks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
