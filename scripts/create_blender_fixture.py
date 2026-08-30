"""Create the deterministic Blender fixture used by integration tests."""

from pathlib import Path
import sys

import bpy
from mathutils import Vector


def main() -> None:
    output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0.75))
    table = bpy.context.object
    table.name = "Table"
    table.scale = (1.5, 1.0, 0.75)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.2, depth=1.0, location=(2, 0, 0.5))
    cup = bpy.context.object
    cup.name = "Cup"

    bpy.ops.object.camera_add(location=(6, -6, 5))
    camera = bpy.context.object
    camera.name = "Camera"
    camera.rotation_euler = (Vector((0, 0, 0.9)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type="AREA", location=(3, -3, 6))
    light = bpy.context.object
    light.name = "KeyLight"
    light.data.energy = 1200
    light.data.shape = "DISK"
    light.data.size = 5

    engines = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256
    bpy.context.scene.render.resolution_percentage = 100
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print("VERIGRAPH_FIXTURE", output)


if __name__ == "__main__":
    main()
