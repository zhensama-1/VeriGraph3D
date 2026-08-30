from __future__ import annotations

import argparse
from enum import Enum
import json
from pathlib import Path

from .dataset import TaskDataset
from .graph import ExecutableSceneGraph
from .graph_visualization import graph_to_dot


def _default(value):
    return value.value if isinstance(value, Enum) else str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a VeriGraph3D task scene graph")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("task_id")
    parser.add_argument("--format", choices=("json", "dot"), default="dot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-false", action="store_true")
    args = parser.parse_args(argv)
    dataset = TaskDataset.load(args.dataset)
    task = next((item for item in dataset.tasks if item.id == args.task_id), None)
    if task is None:
        raise ValueError(f"Task not found: {args.task_id}")
    graph = ExecutableSceneGraph()
    graph.rebuild(dataset.initial_state(task))
    graph.set_goal(task.goal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "dot":
        content = graph_to_dot(graph, args.include_false)
    else:
        content = json.dumps(graph.to_dict(), ensure_ascii=False, indent=2, default=_default)
    args.output.write_text(content, encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

