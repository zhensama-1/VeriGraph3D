"""Generate the deterministic VeriGraph3D research task suite."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


COLORS = [
    [1.0, 0.0, 0.0, 1.0], [0.0, 0.3, 1.0, 1.0],
    [0.1, 0.8, 0.2, 1.0], [1.0, 0.6, 0.0, 1.0],
    [0.6, 0.1, 0.8, 1.0], [0.1, 0.8, 0.8, 1.0],
    [0.9, 0.2, 0.5, 1.0], [0.3, 0.3, 0.3, 1.0],
]


def task(identifier, category, instruction, *, required=None, forbidden=None,
         visual=None, actions=None, uncertainties=None, faults=None, seed=42,
         scene="tabletop"):
    metadata = {"category": category, "difficulty": "mvp", "object_count": 4}
    if faults:
        metadata.update({
            "difficulty": "fault_injection", "faults": faults,
            "expected_failure_type": "constraint_solver_error",
        })
    return {
        "id": identifier, "instruction": instruction, "initial_scene": scene,
        "seed": seed, "required": required or [], "forbidden": forbidden or [],
        "visual": visual or {}, "actions": actions or [],
        "uncertainties": uncertainties or [], "metadata": metadata,
    }


def action(identifier, kind, target, *, parameters=None, reference=None):
    return {
        "id": identifier, "type": kind, "target": target,
        "reference": reference, "parameters": parameters or {},
    }


def build_tasks():
    tasks = []
    move_locations = [
        [3.0, -2.4, 0.5], [3.4, -2.0, 0.5], [2.7, -2.5, 0.5],
        [-3.0, -2.3, 0.5], [-3.4, -2.0, 0.5], [4.0, 2.6, 0.5],
        [-4.0, 2.7, 0.5], [2.8, -2.2, 0.5], [-2.8, -2.2, 0.5],
        [3.6, 0.0, 0.5],
    ]
    for index, location in enumerate(move_locations, 1):
        tasks.append(task(
            f"move_{index:03d}", "move", f"将Cup移动到{location}",
            required=[{"subject": "Cup", "predicate": "location", "expected": location,
                       "tolerance": 0.001}], seed=1000 + index,
        ))

    for index, angle in enumerate([0.25, 0.5, 0.7854, 1.0, 1.5708, 2.0, 2.3562, 3.1416], 1):
        rotation = [0.0, 0.0, angle]
        tasks.append(task(
            f"rotate_{index:03d}", "rotate", f"将Cup绕Z轴旋转到{angle}弧度",
            required=[{"subject": "Cup", "predicate": "rotation", "expected": rotation,
                       "tolerance": 0.001}], seed=2000 + index,
        ))

    for index, factor in enumerate([0.6, 0.75, 0.9, 1.1, 1.25, 1.4, 1.6, 1.8], 1):
        scale = [factor, factor, factor]
        tasks.append(task(
            f"scale_{index:03d}", "scale", f"将Cup缩放为{factor}倍",
            required=[{"subject": "Cup", "predicate": "scale", "expected": scale,
                       "tolerance": 0.001}], seed=3000 + index,
        ))

    uncertainty = [{
        "subject": "Cup", "predicate": "visible_from", "confidence": 0.35,
        "reason": "single_view_occlusion",
    }]
    for index in range(1, 11):
        faults = None
        if index >= 7:
            faults = [{
                "action_id": "action_001", "kind": "z_offset",
                "value": -0.05 * (index - 5), "remaining": 1,
            }]
        tasks.append(task(
            f"place_on_{index:03d}", "place_on", "将Cup无碰撞地放到Table中央",
            required=[["Cup", "on_top_of", "Table"]],
            forbidden=[["Cup", "intersecting", "Table"]],
            uncertainties=uncertainty if index % 2 == 0 else [], faults=faults,
            seed=4000 + index,
        ))

    for index, color in enumerate(COLORS, 1):
        tasks.append(task(
            f"material_{index:03d}", "material", f"修改Cup颜色为RGBA {color}",
            visual={"Cup.color": color}, seed=5000 + index,
        ))

    for index, fov in enumerate([35.0, 45.0, 55.0, 65.0, 75.0], 1):
        tasks.append(task(
            f"camera_{index:03d}", "camera", f"将Camera视场角设为{fov}度",
            required=[{"subject": "Camera", "predicate": "fov_degrees",
                       "expected": fov, "tolerance": 0.001}], seed=6000 + index,
        ))

    for index, energy in enumerate([200.0, 500.0, 800.0, 1200.0, 1800.0], 1):
        tasks.append(task(
            f"light_{index:03d}", "light", f"将Key灯光能量设为{energy}",
            required=[{"subject": "Key", "predicate": "energy",
                       "expected": energy, "tolerance": 0.001}], seed=7000 + index,
        ))

    primitives = ["cube", "sphere", "cylinder", "cone"]
    for index, primitive in enumerate(primitives, 1):
        name = f"Created{index}"
        color = COLORS[index - 1]
        tasks.append(task(
            f"create_{index:03d}", "create", f"创建安全图元{name}",
            required=[{"subject": name, "predicate": "exists", "expected": True}],
            visual={f"{name}.color": color},
            actions=[action(
                "action_001", "create_object", name,
                parameters={
                    "primitive": primitive, "category": "generated",
                    "dimensions": [0.5, 0.5, 0.8],
                    "location": [2.5 + index * 0.3, -2.5, 0.4], "color": color,
                },
            )], seed=8000 + index,
        ))

    for index, target in enumerate(["DeleteA", "DeleteB"], 1):
        tasks.append(task(
            f"delete_{index:03d}", "delete", f"删除授权对象{target}",
            required=[{"subject": target, "predicate": "exists", "expected": False}],
            actions=[action("action_001", "delete_object", target)],
            seed=9000 + index, scene="lifecycle",
        ))

    tasks.extend([
        task(
            "compound_001", "compound", "移动、旋转并缩放Cup",
            required=[
                {"subject": "Cup", "predicate": "location", "expected": [3.4, -2.4, 0.5]},
                {"subject": "Cup", "predicate": "rotation", "expected": [0, 0, 1.2]},
                {"subject": "Cup", "predicate": "scale", "expected": [1.3, 1.3, 1.3]},
            ],
            actions=[
                action("action_001", "move_object", "Cup", parameters={"location": [3.4, -2.4, 0.5]}),
                action("action_002", "rotate_object", "Cup", parameters={"rotation": [0, 0, 1.2]}),
                action("action_003", "scale_object", "Cup", parameters={"scale": [1.3, 1.3, 1.3]}),
            ], seed=10001,
        ),
        task(
            "compound_002", "compound", "放置Cup并修改为红色",
            required=[["Cup", "on_top_of", "Table"]],
            forbidden=[["Cup", "intersecting", "Table"]],
            visual={"Cup.color": COLORS[0]},
            actions=[
                action("action_001", "place_on", "Cup", reference="Table"),
                action("action_002", "set_material", "Cup", parameters={"color": COLORS[0]}),
            ], seed=10002,
        ),
        task(
            "compound_003", "compound", "联合调整相机和灯光",
            required=[
                {"subject": "Camera", "predicate": "fov_degrees", "expected": 60.0},
                {"subject": "Key", "predicate": "energy", "expected": 1400.0},
            ],
            actions=[
                action("action_001", "set_camera", "Camera", parameters={"fov_degrees": 60.0}),
                action("action_002", "set_light", "Key", parameters={"energy": 1400.0}),
            ], seed=10003,
        ),
        task(
            "compound_004", "compound", "创建蓝色圆柱并调整灯光",
            required=[
                {"subject": "GeneratedVase", "predicate": "exists", "expected": True},
                {"subject": "Key", "predicate": "energy", "expected": 1000.0},
            ],
            visual={"GeneratedVase.color": COLORS[1]},
            actions=[
                action("action_001", "create_object", "GeneratedVase", parameters={
                    "primitive": "cylinder", "category": "vase",
                    "dimensions": [0.4, 0.4, 0.9], "location": [3.5, -2.5, 0.45],
                    "color": COLORS[1],
                }),
                action("action_002", "set_light", "Key", parameters={"energy": 1000.0}),
            ], seed=10004,
        ),
    ])
    return tasks


def build_dataset():
    tabletop = {
        "objects": {
            "Table": {"category": "furniture", "location": [0, 0, 0.75],
                      "dimensions": [4, 3, 1.5]},
            "Cup": {"category": "container", "location": [3, -2, 0.5],
                    "dimensions": [0.4, 0.4, 1.0]},
            "Bowl": {"category": "container", "location": [3, 1.9, 0.3],
                     "dimensions": [0.8, 0.8, 0.6]},
            "Block": {"category": "object", "location": [-3, 2, 0.5],
                      "dimensions": [1, 1, 1]},
        },
        "cameras": {"Camera": {"location": [7, -7, 6], "target": [0, 0, 1],
                                  "fov_degrees": 50}},
        "lights": {"Key": {"light_type": "AREA", "location": [3, -3, 6],
                             "energy": 800}},
    }
    lifecycle = json.loads(json.dumps(tabletop))
    lifecycle["objects"]["DeleteA"] = {
        "category": "temporary", "location": [4, 2, 0.4], "dimensions": [0.8, 0.8, 0.8],
    }
    lifecycle["objects"]["DeleteB"] = {
        "category": "temporary", "location": [-4, -2, 0.4], "dimensions": [0.8, 0.8, 0.8],
    }
    tasks = build_tasks()
    return {
        "schema_version": "1.0",
        "dataset": {
            "name": "VeriGraph3D-MVP-64", "version": "1.0",
            "task_count": len(tasks), "seed_policy": "fixed_per_task",
            "category_counts": dict(sorted(Counter(t["metadata"]["category"] for t in tasks).items())),
        },
        "scenes": {"tabletop": tabletop, "lifecycle": lifecycle},
        "tasks": tasks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("examples/tasks_expanded.json"))
    args = parser.parse_args(argv)
    payload = build_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["dataset"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
