"""Ray-trace the actor proxy in Cycles: self-shadows, received shadows and subsurface skin.

    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" -b --factory-startup \\
        --python scripts/render_proxy.py -- --shot <dir> --frame 0 --test uplight

This is the stage that lifts constraints 1-5 in CLAUDE.md, none of which a gain map can reach at
any quality. It renders the proxy mesh from ``make_proxy_mesh.py`` inside the shot's environment
and writes the passes the compositor needs:

    out = plate x R_d x S + G          R_d = diffuse+subsurface ratio, S = shadow catcher, G = specular

``--test`` renders a controlled light instead of the environment, because the environment alone
cannot prove a shadow is being cast. ``uplight`` puts a small source under the chin and ``toplight``
directly above the head: both throw an unmistakable nose shadow if occlusion is working, and no
shadow at all if the mesh is a flat card. That is a far better acceptance test than a metric.

**Blender 5.2.1 traps this stage steps in, all silent:**

* Subsurface is NOT folded into the diffuse passes. There are dedicated
  ``use_pass_subsurface_direct/indirect/color``, and a diffuse-only ratio drops all subsurface
  response without any error.
* ``is_shadow_catcher`` exists on the object INSTANCE, not on ``bpy.types.Object`` -- checking
  the type says False. Verified in ``logs/shadow_catcher_facts.json``.
* ``use_persistent_data = True`` serves stale lighting when stepping frames in one process.
* ``render.fps`` is an int16, so 60000/1001 silently clamps; use fps=60, fps_base=1.001.
* Light objects are invisible to camera rays even with ``visible_camera = True``.
"""
import bpy, sys, json, math, argparse
from pathlib import Path

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument('--shot', default=None)
ap.add_argument('--frame', type=int, default=0)
ap.add_argument('--samples', type=int, default=128)
ap.add_argument('--test', default='', choices=('', 'uplight', 'toplight', 'sidelight'),
                help='controlled light instead of the environment, to prove occlusion')
ap.add_argument('--width', type=int, default=960)
ap.add_argument('--out', default='')
ap.add_argument('--no-cast-shadow', action='store_true',
                help='CONTROL render: the proxy still receives light and shades normally, but '
                     'casts no shadow. The difference against a normal render is the pure '
                     'occlusion contribution, which is the only thing that proves self-shadowing')
args = ap.parse_args(argv)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from shot_paths import Shot                        # stdlib-only

shot = Shot(args.shot)
cfg = shot.config
cam_cfg = cfg.get('camera') or {}
LENS = float(cam_cfg.get('lens_mm', 50.0))
SENSOR = float(cam_cfg.get('sensor_width_mm', 36.0))
DIST = float(cam_cfg.get('actor_distance_m', 1.30))
CAM_H = float(cam_cfg.get('height_m', 1.55))
SRC_W, SRC_H = cfg.get('source_resolution', [1920, 1080])

proxy_npz = shot.relight / 'proxy' / ('%06d.npz' % args.frame)
if not proxy_npz.exists():
    raise SystemExit('No proxy for frame %d. Run scripts/make_proxy_mesh.py --shot %s first.'
                     % (args.frame, shot.name))

# numpy ships inside Blender; the proxy npz is the only thing read here.
import numpy as np
d = np.load(proxy_npz)
verts, faces = d['verts'].astype(float), d['faces'].astype(int)

for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
scn = bpy.context.scene
vl = bpy.context.view_layer
scn.render.engine = 'CYCLES'
scn.cycles.device = 'GPU'
try:
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for dev in prefs.devices:
        dev.use = (dev.type != 'CPU')
except Exception as e:
    print('GPU setup skipped: %s' % e)
scn.cycles.samples = args.samples
scn.cycles.use_denoising = True
# Stale-lighting trap: never reuse BVH/lighting between frames in one process.
scn.render.use_persistent_data = False

# --- proxy mesh ------------------------------------------------------------------------
me = bpy.data.meshes.new('actor_proxy')
me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
me.validate(verbose=False)
me.update()
proxy = bpy.data.objects.new('actor_proxy', me)
scn.collection.objects.link(proxy)
# Smooth shading: the shell is a displaced pixel grid, and flat shading would make every
# quad a visible facet in the shadow terminator.
with bpy.context.temp_override(object=proxy, selected_editable_objects=[proxy]):
    bpy.ops.object.shade_smooth()
if args.no_cast_shadow:
    # Shading is unchanged; only the shadow ray contribution disappears. Subtracting this from a
    # normal render isolates occlusion from dot(n,l), which a threshold on the image alone
    # cannot do -- a shading-only hemisphere passes a naive centroid test with no shadows at all.
    proxy.visible_shadow = False

# --- skin: subsurface by literature ------------------------------------------------------
# Jensen et al. 2001 skin1 scattering radii, mm -> metres. No released model estimates
# subsurface parameters, so these are the documented literature values, not a guess.
JENSEN_SKIN1_RADIUS_MM = (3.67, 1.37, 0.68)
mat = bpy.data.materials.new('proxy_skin')
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.63, 0.44, 0.34, 1.0)     # mid skin albedo
bsdf.inputs['Roughness'].default_value = 0.45
if 'Specular IOR Level' in bsdf.inputs:
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
sss_set = {}
try:
    bsdf.subsurface_method = 'RANDOM_WALK_SKIN'
    sss_set['method'] = 'RANDOM_WALK_SKIN'
except Exception as e:
    sss_set['method_error'] = str(e)
for name, val in (('Subsurface Weight', 0.35),
                  ('Subsurface Radius', tuple(r / 1000.0 for r in JENSEN_SKIN1_RADIUS_MM)),
                  ('Subsurface Scale', 1.0)):
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val
        sss_set[name] = val
proxy.data.materials.append(mat)

# --- camera ------------------------------------------------------------------------------
cam_data = bpy.data.cameras.new('Plate camera')
cam_data.lens = LENS
cam_data.sensor_width = SENSOR
cam_data.sensor_fit = 'HORIZONTAL'
cam = bpy.data.objects.new('Plate camera', cam_data)
scn.collection.objects.link(cam)
cam.location = (0.0, -DIST, CAM_H)
cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)
scn.camera = cam

# --- lighting ----------------------------------------------------------------------------
world = bpy.data.worlds.new('w')
scn.world = world
wnt = world.node_tree
bg = next(n for n in wnt.nodes if n.type == 'BACKGROUND')
lights = []
if args.test:
    # A controlled source, in an otherwise black world, so any shadow seen is cast by the
    # proxy and nothing else. Emissive MESH, not a light object: light objects are invisible
    # to camera rays, which was a round-1 trap.
    bg.inputs['Strength'].default_value = 0.0
    # Head is near the top of the matte; place relative to the proxy's own bounding box so
    # this works on any shot rather than only this one.
    zs, ys, xs = verts[:, 2], verts[:, 1], verts[:, 0]
    head_z, chin_z = float(zs.max()), float(np.percentile(zs, 70))
    cx, front_y = float(np.median(xs)), float(ys.min())
    place = {'uplight':   (cx, front_y - 0.25, chin_z - 0.30),
             'toplight':  (cx, front_y - 0.15, head_z + 0.30),
             'sidelight': (cx - 0.55, front_y - 0.20, (head_z + chin_z) / 2)}[args.test]
    lm = bpy.data.meshes.new('src')
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.06)   # 12 cm source: hard
    bm.to_mesh(lm); bm.free()
    lob = bpy.data.objects.new('test_source', lm)
    scn.collection.objects.link(lob)
    lob.location = place
    # Aim the emitter at the head.
    tgt = np.array([cx, 0.0, (head_z + chin_z) / 2]) - np.array(place)
    lob.rotation_euler = (math.atan2(math.hypot(tgt[0], tgt[1]), tgt[2]), 0.0,
                          math.atan2(tgt[1], tgt[0]) + math.pi / 2)
    lmat = bpy.data.materials.new('emit')
    lmat.use_nodes = True
    lnt = lmat.node_tree
    for n in list(lnt.nodes):
        if n.type != 'OUTPUT_MATERIAL':
            lnt.nodes.remove(n)
    em = lnt.nodes.new('ShaderNodeEmission')
    em.inputs['Strength'].default_value = 120.0
    lnt.links.new(em.outputs['Emission'], lnt.nodes['Material Output'].inputs['Surface'])
    lob.data.materials.append(lmat)
    lights.append({'test': args.test, 'location': list(place), 'size_m': 0.12,
                   'emission_strength': 120.0})
else:
    hdri = ROOT / 'assets/hdri/warm_restaurant_night_4k.exr'
    if not hdri.exists():
        raise SystemExit('Missing %s' % hdri)
    env = wnt.nodes.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(str(hdri))
    wnt.links.new(env.outputs['Color'], bg.inputs['Color'])
    bg.inputs['Strength'].default_value = float(
        (cfg.get('environment') or {}).get('hdri_strength', 1.0))
    lights.append({'environment': hdri.name})

# --- passes ------------------------------------------------------------------------------
# Subsurface is NOT in the diffuse passes in 5.2.1. Enabling them explicitly is the whole
# reason this block is not two lines.
wanted = {'use_pass_combined': True, 'use_pass_z': True, 'use_pass_normal': True,
          'use_pass_diffuse_direct': True, 'use_pass_diffuse_indirect': True,
          'use_pass_diffuse_color': True,
          'use_pass_glossy_direct': True, 'use_pass_glossy_indirect': True,
          'use_pass_glossy_color': True,
          'use_pass_subsurface_direct': True, 'use_pass_subsurface_indirect': True,
          'use_pass_subsurface_color': True,
          'use_pass_shadow_catcher': True}
pass_status = {}
for name, val in wanted.items():
    holder = vl if hasattr(vl, name) else getattr(vl, 'cycles', None)
    if holder is not None and hasattr(holder, name):
        setattr(holder, name, val)
        pass_status[name] = True
    else:
        pass_status[name] = 'ABSENT in this Blender build'

W = args.width
scn.render.resolution_x = W
scn.render.resolution_y = int(round(W * SRC_H / SRC_W))
scn.render.resolution_percentage = 100
# media_type gates which file_format values are ACCEPTED, while the enum still *lists* all of
# them -- so 'OPEN_EXR_MULTILAYER' raises "not found in (...)" under media_type 'IMAGE' even
# though it appears in that very tuple. Set media_type first, and to MULTI_LAYER_IMAGE.
scn.render.image_settings.media_type = 'MULTI_LAYER_IMAGE'
scn.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
scn.render.image_settings.color_depth = '32'
scn.render.film_transparent = True

out_dir = Path(args.out) if args.out else (shot.relight / 'proxy_render')
out_dir.mkdir(parents=True, exist_ok=True)
tag = args.test or 'environment'
tag = tag + ('_noshadow' if args.no_cast_shadow else '')
scn.render.filepath = str(out_dir / ('%06d_%s' % (args.frame, tag)))
bpy.ops.render.render(write_still=True)

# A viewable preview of the combined pass. Multilayer EXR is unreadable by opencv (it returns
# None with no error), so the acceptance test has to be written from inside Blender.
preview = str(out_dir / ('%06d_%s_preview.png' % (args.frame, tag)))
try:
    rr = bpy.data.images['Render Result']
    st = scn.render.image_settings
    st.media_type = 'IMAGE'
    st.file_format = 'PNG'
    st.color_depth = '8'
    st.color_mode = 'RGBA'
    rr.save_render(filepath=preview, scene=scn)
except Exception as e:
    preview = 'FAILED: %s' % e

report = {'shot': shot.name, 'frame': args.frame, 'mode': tag, 'preview': preview,
          'blender': bpy.app.version_string,
          'proxy': {'verts': int(len(verts)), 'faces': int(len(faces)),
                    'extent_m': (verts.max(0) - verts.min(0)).round(4).tolist()},
          'subsurface': sss_set,
          'jensen_2001_skin1_radius_mm': list(JENSEN_SKIN1_RADIUS_MM),
          'passes': pass_status,
          'lighting': lights,
          'samples': args.samples,
          'resolution': [scn.render.resolution_x, scn.render.resolution_y],
          'output': scn.render.filepath + '.exr'}
(shot.logs).mkdir(parents=True, exist_ok=True)
shot.log('proxy_render_%s_%06d.json' % (tag, args.frame)).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
