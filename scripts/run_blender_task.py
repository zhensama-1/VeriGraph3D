"""Run one VeriGraph3D task inside Blender.

Example:
  blender --background scene.blend --python scripts/run_blender_task.py -- \
    examples/tasks.json place_and_color_001 --output blender_run.json
"""

from __future__ import annotations

import argparse
from enum import Enum
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from verigraph3d.agent import AgentConfig, VeriGraph3DAgent  # noqa: E402
from verigraph3d.backends import BlenderBackend  # noqa: E402
from verigraph3d.agentblender import (  # noqa: E402
    AgentBlenderInstalledReader,
    AgentBlenderRuntimeBridge,
    AgentBlenderWorldStateBackend,
)
from verigraph3d.dataset import TaskDataset  # noqa: E402
from verigraph3d.visual import EnsembleVisualVerifier, ReferenceImageVerifier  # noqa: E402
from verigraph3d.vlm import VLMTaskInterpreter, VLMVisualVerifier, create_vlm_client  # noqa: E402


def _arguments() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("task_id")
    parser.add_argument("--output", type=Path, default=Path("blender_run.json"))
    parser.add_argument("--render-directory", default="blender_renders")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vlm-task-understanding", action="store_true")
    parser.add_argument("--vlm-visual-verification", action="store_true")
    parser.add_argument(
        "--agentblender-runtime-factory",
        help="ABWS runtime factory in module:callable form; enables transactional world state",
    )
    parser.add_argument(
        "--agentblender-src",
        type=Path,
        help="Optional directory containing the installed/developed agent_blender package",
    )
    parser.add_argument(
        "--agentblender-world-state",
        action="store_true",
        help="Read authoritative state through ABWS and execute with BlenderBackend",
    )
    return parser.parse_args(raw)


def _default(value):
    if isinstance(value, Enum):
        return value.value
    return str(value)


def main() -> int:
    args = _arguments()
    dataset = TaskDataset.load(args.dataset)
    task = next((item for item in dataset.tasks if item.id == args.task_id), None)
    if task is None:
        raise ValueError(f"Task not found: {args.task_id}")
    blender_backend = BlenderBackend()
    backend = blender_backend
    if args.agentblender_src:
        sys.path.insert(0, str(args.agentblender_src.resolve()))
    if args.agentblender_runtime_factory:
        bridge = AgentBlenderRuntimeBridge.from_factory(args.agentblender_runtime_factory)
        backend = bridge.backend(capability_backend=blender_backend)
    elif args.agentblender_world_state:
        backend = AgentBlenderWorldStateBackend(
            blender_backend, AgentBlenderInstalledReader()
        )
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        backend.save_checkpoint(str(args.checkpoint.resolve()))
    client = create_vlm_client() if args.vlm_task_understanding or args.vlm_visual_verification else None
    interpreter = VLMTaskInterpreter(client) if args.vlm_task_understanding else None
    visual = (
        EnsembleVisualVerifier([ReferenceImageVerifier(), VLMVisualVerifier(client)], [0.4, 0.6])
        if args.vlm_visual_verification else ReferenceImageVerifier()
    )
    config = AgentConfig(
        render_after_action=bool(task.reference_images) or args.vlm_visual_verification,
        render_directory=args.render_directory,
    )
    agent = VeriGraph3DAgent(backend, config=config, visual_verifier=visual, task_interpreter=interpreter)
    supplied_goal = None if args.vlm_task_understanding else task.goal
    result = agent.run(
        instruction=task.instruction, goal=supplied_goal,
        reference_images=task.reference_images, uncertainties=task.uncertainties,
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")
    print(json.dumps({"accepted": result["accepted"], "decision": result["decision"], "metrics": result["metrics"]}, ensure_ascii=False))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
