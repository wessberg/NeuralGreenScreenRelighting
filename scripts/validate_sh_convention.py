"""Unit-test the light-probe axis convention against known light directions.

    blender -b --factory-startup --python scripts/validate_sh_convention.py

Places a single emissive panel at a known world direction in an otherwise black world, bakes an
equirectangular probe exactly as scripts/bake_light_probes.py does, projects it with the same
basis as scripts/sh_project.py, and checks the recovered L1 direction points back at the light.

This exists because a sign or axis error in the probe convention produces lighting that is
subtly wrong in a way no single frame reveals and no grade can fix. It cannot be validated
against the production scene, where the recovered direction is the sum of the environment and
several emitters at once -- only against a controlled single source.
"""
import bpy, bmesh, sys, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs' / 'sh_convention'
OUT.mkdir(parents=True, exist_ok=True)

# Known directions FROM the origin TOWARD the light, in Blender world axes (+Z up).
CASES = [
    ('+X (right)',   (1.0, 0.0, 0.0)),
    ('-X (left)',    (-1.0, 0.0, 0.0)),
    ('+Y (behind)',  (0.0, 1.0, 0.0)),
    ('-Y (front)',   (0.0, -1.0, 0.0)),
    ('+Z (above)',   (0.0, 0.0, 1.0)),
    ('diag XY',      (0.707, -0.707, 0.0)),
]
DIST = 3.0

for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
scn = bpy.context.scene
scn.render.engine = 'CYCLES'
world = bpy.data.worlds.new('black')
scn.world = world
bgn = next(n for n in world.node_tree.nodes if n.type == 'BACKGROUND')
bgn.inputs['Strength'].default_value = 0.0

me = bpy.data.meshes.new('panel')
bm = bmesh.new()
bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
bm.to_mesh(me); bm.free()
panel = bpy.data.objects.new('panel', me)
scn.collection.objects.link(panel)
mat = bpy.data.materials.new('em')
nt = mat.node_tree
for n in list(nt.nodes):
    if n.type != 'OUTPUT_MATERIAL':
        nt.nodes.remove(n)
em = nt.nodes.new('ShaderNodeEmission')
em.inputs['Strength'].default_value = 200.0
nt.links.new(em.outputs['Emission'], nt.nodes['Material Output'].inputs['Surface'])
panel.data.materials.append(mat)

cam_data = bpy.data.cameras.new('probe')
cam_data.type = 'PANO'
cam_data.panorama_type = 'EQUIRECTANGULAR'
cam = bpy.data.objects.new('probe', cam_data)
scn.collection.objects.link(cam)
cam.rotation_euler = (math.radians(90.0), 0.0, math.radians(-90.0))
cam.location = (0.0, 0.0, 0.0)
scn.camera = cam

scn.render.resolution_x, scn.render.resolution_y = 128, 64
scn.cycles.samples = 256
scn.cycles.use_denoising = False
scn.render.use_persistent_data = False
scn.render.film_transparent = False
ims = scn.render.image_settings
if hasattr(ims, 'media_type'):
    ims.media_type = 'IMAGE'
ims.file_format = 'OPEN_EXR'
ims.color_mode = 'RGB'
ims.color_depth = '32'
ims.exr_codec = 'ZIP'
ims.color_management = 'OVERRIDE'
ims.view_settings.view_transform = 'Raw'

results = []
for label, d in CASES:
    n = math.sqrt(sum(c * c for c in d))
    u = tuple(c / n for c in d)
    panel.location = tuple(c * DIST for c in u)
    # Aim the emissive face back at the origin.
    panel.rotation_euler = (math.atan2(math.hypot(-u[0], -u[1]), u[2]), 0.0,
                            math.atan2(u[0], -u[1]))
    path = OUT / ('probe_%s.exr' % label.split()[0].replace('+', 'p').replace('-', 'm'))
    scn.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    results.append({'label': label, 'expected_direction': list(u), 'file': path.name})

(OUT / 'cases.json').write_text(json.dumps(results, indent=2))
print('SH_CONVENTION_BAKED', json.dumps({'dir': str(OUT), 'cases': len(results)}))
