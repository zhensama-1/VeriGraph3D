"""Generate the 24-task stratified real-Blender evaluation subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def constraint(subject, predicate, expected, tolerance=0.001):
    return {
        "subject": subject, "predicate": predicate,
        "expected": expected, "tolerance": tolerance,
    }


def action(identifier, kind, target, parameters=None, reference=None):
    return {
        "id": identifier, "type": kind, "target": target,
        "reference": reference, "parameters": parameters or {},
    }


def task(identifier, category, instruction, *, required=None, forbidden=None,
         visual=None, actions=None, seed=42):
    return {
        "id": identifier, "instruction": instruction, "initial_scene": {},
        "seed": seed, "required": required or [], "forbidden": forbidden or [],
        "visual": visual or {}, "actions": actions or [],
        "metadata": {
            "category": category, "backend": "agentblender_atomic",
            "source_scene": "abws_test_fixture",
        },
    }


def build_tasks():
    chair = "ABWS__chair_01"
    desk = "ABWS__desk_01"
    camera = "ABWS__camera_main"
    light = "ABWS__key_light"
    tasks = []
    for index, location in enumerate([
        [1.5, -1.0, 0.455], [2.5, -1.8, 0.455],
        [1.5, -2.0, 0.455], [2.8, 0.0, 0.455],
    ], 1):
        tasks.append(task(
            f"blender_move_{index:02d}", "move", f"移动椅子到{location}",
            required=[constraint(chair, "location", location)], seed=1100 + index,
        ))
    for index, angle in enumerate([0.5, 1.0, 1.5708], 1):
        tasks.append(task(
            f"blender_rotate_{index:02d}", "rotate", f"旋转椅子到{angle}弧度",
            required=[constraint(chair, "rotation", [0, 0, angle])], seed=1200 + index,
        ))
    for index, factor in enumerate([0.8, 1.1, 1.25], 1):
        tasks.append(task(
            f"blender_scale_{index:02d}", "scale", f"缩放椅子为{factor}倍",
            required=[constraint(chair, "scale", [factor, factor, factor])], seed=1300 + index,
        ))
    for index in range(1, 5):
        tasks.append(task(
            f"blender_place_{index:02d}", "place_on", "将椅子无碰撞放到桌面",
            required=[[chair, "on_top_of", desk]],
            forbidden=[[chair, "intersecting", desk]], seed=1400 + index,
        ))
    colors = [
        [1.0, 0.0, 0.0, 1.0], [0.0, 0.3, 1.0, 1.0], [0.1, 0.8, 0.2, 1.0],
    ]
    for index, color in enumerate(colors, 1):
        tasks.append(task(
            f"blender_material_{index:02d}", "material", f"修改椅子颜色为{color}",
            visual={f"{chair}.color": color}, seed=1500 + index,
        ))
    for index, fov in enumerate([45.0, 65.0], 1):
        tasks.append(task(
            f"blender_camera_{index:02d}", "camera", f"设置相机视场角为{fov}",
            required=[constraint(camera, "fov_degrees", fov)], seed=1600 + index,
        ))
    for index, energy in enumerate([600.0, 1400.0], 1):
        tasks.append(task(
            f"blender_light_{index:02d}", "light", f"设置主灯能量为{energy}",
            required=[constraint(light, "energy", energy)], seed=1700 + index,
        ))
    tasks.append(task(
        "blender_create_01", "create", "创建蓝色圆柱花瓶",
        required=[constraint("ABWS__vase_01", "exists", True)],
        visual={"ABWS__vase_01.color": colors[1]},
        actions=[action("action_001", "create_object", "vase_01", {
            "primitive": "cylinder", "category": "vase",
            "dimensions": [0.4, 0.4, 0.8], "location": [1.0, -2.0, 0.4],
            "color": colors[1],
        })], seed=1801,
    ))
    tasks.append(task(
        "blender_delete_01", "delete", "删除授权临时对象",
        required=[constraint("ABWS__temp_01", "exists", False)],
        actions=[action("action_001", "delete_object", "ABWS__temp_01")], seed=1901,
    ))
    tasks.append(task(
        "blender_compound_01", "compound", "移动、旋转并修改椅子颜色",
        required=[
            constraint(chair, "location", [1.6, -1.8, 0.455]),
            constraint(chair, "rotation", [0, 0, 0.8]),
        ],
        visual={f"{chair}.color": colors[0]},
        actions=[
            action("action_001", "move_object", chair, {"location": [1.6, -1.8, 0.455]}),
            action("action_002", "rotate_object", chair, {"rotation": [0, 0, 0.8]}),
            action("action_003", "set_material", chair, {"color": colors[0]}),
        ], seed=2001,
    ))
    return tasks


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("examples/blender_tasks_expanded.json"))
    args = parser.parse_args(argv)
    tasks = build_tasks()
    payload = {
        "schema_version": "1.0",
        "dataset": {
            "name": "VeriGraph3D-Blender-24", "version": "1.0",
            "task_count": len(tasks),
            "category_counts": dict(sorted(Counter(t["metadata"]["category"] for t in tasks).items())),
        },
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["dataset"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
