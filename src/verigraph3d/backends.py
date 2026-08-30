from __future__ import annotations

from dataclasses import dataclass, replace

from .models import ActionType, CameraState, LightState, SceneBackend, SceneObject, SceneState, SemanticAction


class MemoryBackend(SceneBackend):
    """Deterministic backend used by tests and headless experiments."""

    def __init__(self, state: SceneState | None = None) -> None:
        self.state = (state or SceneState()).clone()

    def read_state(self) -> SceneState:
        return self.state.clone()

    def restore(self, state: SceneState) -> None:
        self.state = state.clone()

    def apply(self, action: SemanticAction) -> None:
        p = action.parameters
        if action.type == ActionType.CREATE_OBJECT:
            fields = {
                key: value for key, value in p.items()
                if key in {
                    "category", "location", "rotation", "scale", "dimensions",
                    "color", "material", "visible", "parent", "rigid_body",
                    "collision_enabled", "metadata",
                }
            }
            for key in ("location", "rotation", "scale", "dimensions", "color"):
                if key in fields:
                    fields[key] = tuple(fields[key])
            self.state.objects[action.target] = SceneObject(name=action.target, **fields)
        elif action.type == ActionType.DELETE_OBJECT:
            del self.state.objects[action.target]
        elif action.type in {ActionType.MOVE_OBJECT, ActionType.PLACE_ON, ActionType.PLACE_INSIDE, ActionType.ALIGN}:
            self.state.objects[action.target].location = tuple(p["location"])
        elif action.type == ActionType.ROTATE_OBJECT:
            self.state.objects[action.target].rotation = tuple(p["rotation"])
        elif action.type == ActionType.SCALE_OBJECT:
            self.state.objects[action.target].scale = tuple(p["scale"])
        elif action.type == ActionType.SET_MATERIAL:
            obj = self.state.objects[action.target]
            if "material" in p:
                obj.material = p["material"]
            if "color" in p:
                obj.color = tuple(p["color"])
        elif action.type == ActionType.SET_CAMERA:
            camera = self.state.cameras.setdefault(action.target, CameraState(name=action.target))
            for key in ("location", "target", "fov_degrees"):
                if key in p:
                    setattr(camera, key, tuple(p[key]) if key != "fov_degrees" else float(p[key]))
        elif action.type == ActionType.SET_LIGHT:
            light = self.state.lights.setdefault(action.target, LightState(name=action.target))
            for key in ("location", "color", "energy", "visible"):
                if key in p:
                    setattr(light, key, tuple(p[key]) if key in {"location", "color"} else p[key])
        else:
            raise NotImplementedError(action.type.value)
        self.state.timestamp += 1


@dataclass(slots=True)
class FaultRule:
    """A reproducible execution fault used for recovery experiments."""

    action_id: str
    kind: str
    value: float | None = None
    remaining: int = 1


class FaultInjectingBackend(SceneBackend):
    """Wraps a backend and injects declared faults without changing production code."""

    def __init__(self, backend: SceneBackend, rules: list[FaultRule]) -> None:
        self.backend = backend
        self.rules = rules

    def read_state(self) -> SceneState:
        return self.backend.read_state()

    def restore(self, state: SceneState) -> None:
        self.backend.restore(state)

    def apply(self, action: SemanticAction) -> None:
        rule = next((r for r in self.rules if r.action_id == action.id and r.remaining > 0), None)
        if rule is None:
            self.backend.apply(action)
            return
        rule.remaining -= 1
        if rule.kind == "execution_error":
            raise RuntimeError("Injected execution error")
        if rule.kind == "no_op":
            return
        if rule.kind == "z_offset":
            if "location" not in action.parameters:
                raise ValueError("z_offset fault requires a solved location")
            location = list(action.parameters["location"])
            location[2] += float(rule.value or 0.0)
            action = replace(action, parameters={**action.parameters, "location": tuple(location)})
            self.backend.apply(action)
            return
        raise ValueError(f"Unsupported fault kind: {rule.kind}")


class BlenderBackend(SceneBackend):
    """Optional bpy adapter. Importing the package does not require Blender."""

    def __init__(self) -> None:
        try:
            import bpy  # type: ignore
        except ImportError as exc:
            raise RuntimeError("BlenderBackend must run inside Blender's Python environment") from exc
        self.bpy = bpy

    def read_state(self) -> SceneState:
        objects: dict[str, SceneObject] = {}
        cameras: dict[str, CameraState] = {}
        lights: dict[str, LightState] = {}
        for obj in self.bpy.context.scene.objects:
            if obj.type == "CAMERA":
                cameras[obj.name] = CameraState(
                    obj.name, tuple(obj.location), (0, 0, 0), obj.data.angle * 57.2958,
                    {"abws_id": obj.get("abws_id")},
                )
                continue
            if obj.type == "LIGHT":
                lights[obj.name] = LightState(
                    obj.name, obj.data.type, tuple(obj.location), tuple(obj.data.color),
                    float(obj.data.energy), not obj.hide_render,
                    {"abws_id": obj.get("abws_id")},
                )
                continue
            if obj.type not in {"MESH", "EMPTY"}:
                continue
            color = tuple(obj.color)
            material = obj.active_material.name if obj.active_material else None
            world_corners = [obj.matrix_world @ self._vector(corner) for corner in obj.bound_box]
            world_minimum = tuple(min(corner[i] for corner in world_corners) for i in range(3))
            world_maximum = tuple(max(corner[i] for corner in world_corners) for i in range(3))
            objects[obj.name] = SceneObject(
                name=obj.name, category=obj.type.lower(), location=tuple(obj.location),
                rotation=tuple(obj.rotation_euler), scale=tuple(obj.scale), dimensions=tuple(obj.dimensions),
                color=color, material=material, visible=not obj.hide_render,
                parent=obj.parent.name if obj.parent else None, rigid_body=obj.rigid_body is not None,
                metadata={"world_aabb": {"minimum": world_minimum, "maximum": world_maximum}, "aabb_origin": tuple(obj.location)},
            )
        for name, scene_object in objects.items():
            blender_object = self.bpy.data.objects.get(name)
            if blender_object:
                scene_object.metadata["camera_visibility"] = {
                    camera_name: self._visibility_score(blender_object, self.bpy.data.objects.get(camera_name))
                    for camera_name in cameras
                }
        return SceneState(objects=objects, cameras=cameras, lights=lights, timestamp=int(self.bpy.context.scene.frame_current))

    @staticmethod
    def _vector(values):
        from mathutils import Vector  # type: ignore
        return Vector(values)

    def _visibility_score(self, target, camera) -> float:
        if camera is None or camera.type != "CAMERA" or target.hide_render:
            return 0.0
        from bpy_extras.object_utils import world_to_camera_view  # type: ignore

        scene = self.bpy.context.scene
        depsgraph = self.bpy.context.evaluated_depsgraph_get()
        corners = [target.matrix_world @ self._vector(corner) for corner in target.bound_box]
        center = sum(corners, self._vector((0, 0, 0))) / len(corners)
        samples = [center, *corners]
        origin = camera.matrix_world.translation
        visible = 0
        for point in samples:
            projected = world_to_camera_view(scene, camera, point)
            if projected.z <= 0 or not (0 <= projected.x <= 1 and 0 <= projected.y <= 1):
                continue
            ray = point - origin
            distance = ray.length
            if distance <= 1e-6:
                visible += 1
                continue
            hit, _, _, _, hit_object, _ = scene.ray_cast(
                depsgraph, origin, ray.normalized(), distance=max(0.0, distance - 1e-4)
            )
            if not hit or (hit_object and hit_object.original.name == target.name):
                visible += 1
        return visible / len(samples)

    def score_view(self, camera: CameraState, subjects: set[str]) -> float:
        """Evaluate a candidate camera against Blender geometry without rendering."""
        active = self.bpy.context.scene.camera
        if active is None:
            active = next((obj for obj in self.bpy.context.scene.objects if obj.type == "CAMERA"), None)
        if active is None:
            return 0.0
        matrix, angle = active.matrix_world.copy(), active.data.angle
        try:
            active.location = camera.location
            direction = self._vector(camera.target) - active.location
            active.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
            import math
            active.data.angle = math.radians(camera.fov_degrees)
            self.bpy.context.view_layer.update()
            scores = [
                self._visibility_score(obj, active)
                for name in subjects
                for obj in [self.bpy.data.objects.get(name)]
                if obj is not None
            ]
            return sum(scores) / len(scores) if scores else 0.0
        finally:
            active.matrix_world = matrix
            active.data.angle = angle
            self.bpy.context.view_layer.update()

    def apply(self, action: SemanticAction) -> None:
        p = action.parameters
        if action.type == ActionType.CREATE_OBJECT:
            self._create_object(action.target, p)
            self.bpy.context.view_layer.update()
            return
        obj = self.bpy.data.objects.get(action.target)
        if action.type == ActionType.DELETE_OBJECT and obj:
            self.bpy.data.objects.remove(obj, do_unlink=True)
            return
        if obj is None:
            raise ValueError(f"Blender object not found: {action.target}")
        if "location" in p:
            obj.location = p["location"]
        if "rotation" in p:
            obj.rotation_euler = p["rotation"]
        if "scale" in p:
            obj.scale = p["scale"]
        if action.type == ActionType.SET_MATERIAL:
            self._set_material(obj, p)
        if action.type == ActionType.SET_CAMERA and obj.type == "CAMERA" and "fov_degrees" in p:
            import math
            obj.data.angle = math.radians(float(p["fov_degrees"]))
        if action.type == ActionType.SET_CAMERA and obj.type == "CAMERA" and "target" in p:
            direction = self._vector(p["target"]) - obj.location
            obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        if action.type == ActionType.SET_LIGHT and obj.type == "LIGHT":
            for key in ("energy", "color"):
                if key in p:
                    setattr(obj.data, key, p[key])
        self.bpy.context.view_layer.update()

    def restore(self, state: SceneState) -> None:
        expected = set(state.objects) | set(state.cameras) | set(state.lights)
        for obj in list(self.bpy.context.scene.objects):
            if obj.name not in expected and obj.type in {"MESH", "EMPTY", "CAMERA"}:
                self.bpy.data.objects.remove(obj, do_unlink=True)
        for name, snapshot in state.objects.items():
            obj = self.bpy.data.objects.get(name)
            if obj:
                obj.location, obj.rotation_euler, obj.scale = snapshot.location, snapshot.rotation, snapshot.scale
                obj.hide_render = not snapshot.visible
                self._set_material(obj, {"material": snapshot.material, "color": snapshot.color})
        for name, snapshot in state.cameras.items():
            obj = self.bpy.data.objects.get(name)
            if obj:
                obj.location = snapshot.location
        for name, snapshot in state.lights.items():
            obj = self.bpy.data.objects.get(name)
            if obj:
                obj.location = snapshot.location
                obj.data.color = snapshot.color
                obj.data.energy = snapshot.energy
                obj.hide_render = not snapshot.visible
        self.bpy.context.view_layer.update()

    def save_checkpoint(self, path: str) -> None:
        """Save a copy without replacing the currently opened .blend file."""
        self.bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)

    def render(self, path: str, camera_name: str | None = None) -> str:
        scene = self.bpy.context.scene
        if camera_name:
            camera = self.bpy.data.objects.get(camera_name)
            if camera is None or camera.type != "CAMERA":
                raise ValueError(f"Camera not found: {camera_name}")
            scene.camera = camera
        scene.render.filepath = path
        self.bpy.ops.render.render(write_still=True)
        return path

    def _create_object(self, name: str, parameters: dict) -> None:
        primitive = str(parameters.get("primitive", "cube")).lower()
        operators = {
            "cube": self.bpy.ops.mesh.primitive_cube_add,
            "sphere": self.bpy.ops.mesh.primitive_uv_sphere_add,
            "cylinder": self.bpy.ops.mesh.primitive_cylinder_add,
            "cone": self.bpy.ops.mesh.primitive_cone_add,
            "plane": self.bpy.ops.mesh.primitive_plane_add,
        }
        if primitive not in operators:
            raise ValueError(f"Unsupported primitive: {primitive}")
        operators[primitive](location=parameters.get("location", (0, 0, 0)))
        obj = self.bpy.context.object
        obj.name = name
        if "rotation" in parameters:
            obj.rotation_euler = parameters["rotation"]
        if "scale" in parameters:
            obj.scale = parameters["scale"]
        if "color" in parameters or "material" in parameters:
            self._set_material(obj, parameters)

    def _set_material(self, obj, parameters: dict) -> None:
        material_name = parameters.get("material") or f"{obj.name}_Material"
        material = self.bpy.data.materials.get(material_name) or self.bpy.data.materials.new(material_name)
        if "color" in parameters:
            color = tuple(parameters["color"])
            rgba = color if len(color) == 4 else (*color, 1.0)
            material.diffuse_color = rgba
            material.use_nodes = True
            principled = material.node_tree.nodes.get("Principled BSDF") if material.node_tree else None
            if principled and "Base Color" in principled.inputs:
                principled.inputs["Base Color"].default_value = rgba
            if principled and "Alpha" in principled.inputs:
                principled.inputs["Alpha"].default_value = rgba[3]
            obj.color = material.diffuse_color
        if not obj.data or not hasattr(obj.data, "materials"):
            return
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)
