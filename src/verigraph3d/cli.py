from __future__ import annotations

import argparse
from enum import Enum
import json
from pathlib import Path

from .agent import VeriGraph3DAgent
from .backends import MemoryBackend
from .models import CameraState, SceneObject, SceneState


def demo_scene() -> SceneState:
    return SceneState(
        objects={
            "Table": SceneObject("Table", "furniture", (0, 0, 0.75), dimensions=(3, 2, 1.5), metadata={"allow_floating": True}),
            "Cup": SceneObject("Cup", "container", (2, 0, 0.5), dimensions=(0.4, 0.4, 1.0)),
        },
        cameras={"Camera": CameraState()},
    )


def _json_default(value):
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VeriGraph3D research prototype")
    parser.add_argument("--instruction", default="把Cup放到Table上，并把Cup改成红色")
    parser.add_argument("--output", type=Path, help="Write the complete experiment trace as JSON")
    args = parser.parse_args(argv)
    result = VeriGraph3DAgent(MemoryBackend(demo_scene())).run(instruction=args.instruction)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, default=_json_default)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

