from __future__ import annotations

import argparse
from enum import Enum
import json
from pathlib import Path
import sys
import tempfile

from .agent import VeriGraph3DAgent
from .dataset import TaskDataset


def _json_default(value):
    if isinstance(value, Enum):
        return value.value
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run VeriGraph3D through ABWS candidate .blend transactions"
    )
    parser.add_argument("blend_file", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("task_id")
    parser.add_argument("--output-blend", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--abws-src",
        type=Path,
        default=Path("external/AgentBlender-World-State/src"),
    )
    parser.add_argument("--blender-executable", default=r"E:\Blender\blender.exe")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.abws_src.resolve()))
    from .agentblender import AgentBlenderAtomicBlendBackend

    dataset = TaskDataset.load(args.dataset)
    task = next((item for item in dataset.tasks if item.id == args.task_id), None)
    if task is None:
        raise ValueError(f"Task not found: {args.task_id}")
    with tempfile.TemporaryDirectory(prefix="verigraph3d_abws_") as work_directory:
        backend = AgentBlenderAtomicBlendBackend(
            args.blend_file,
            work_directory,
            blender_executable=args.blender_executable,
        )
        result = VeriGraph3DAgent(backend).run(
            instruction=task.instruction,
            goal=task.goal,
            actions=task.actions or None,
            reference_images=task.reference_images,
            uncertainties=task.uncertainties,
        )
        if result["accepted"]:
            backend.export(args.output_blend)
        report = args.report or args.output_blend.with_suffix(".json")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
    print(json.dumps({
        "accepted": result["accepted"],
        "decision": result["decision"],
        "output_blend": str(args.output_blend.resolve()) if result["accepted"] else None,
        "report": str(report.resolve()),
    }, ensure_ascii=False))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
