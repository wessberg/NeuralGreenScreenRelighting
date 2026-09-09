"""Render the SCENE's own corneal reflection, per frame, at the plate's own resolution.

    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" -b --factory-startup \\
        --python scripts/render_eye_catchlight.py -- --shot <dir>

Round 33 removed the capture room's catchlight and put nothing back, so a lit eye became a
flat brown disc. That is not a temporal-stability problem and no amount of tuning the
removal fixes it: the artist is looking at an eye with no specular in it at all.

This is the other half, and it is deliberately the same construction stage 11d already
validated (`stress_proxy.py --catchlight`, `validate_catchlight.py`): **put a mirror where
the cornea is and let Cycles find the source.** Presence, size, shape, colour and occlusion
then all come from the scene rather than from a rule. What is different here is only the
scene: `stress_proxy.py` lights the corneas with its own lightbulb tour, which is a demo,
while this loads the shot's actual environment out of `scene_library/<env>` -- the same
world node graph `build_hdri_scene.py` builds and the same one the background render used,
so the panorama orientation is shared with an already-rendered path rather than retyped.

Three things that are easy to get wrong and are not settings:

* **The cornea is a dielectric, not a mirror.** IOR 1.376, black base colour, so Fresnel
  gives about 2.5% at normal incidence. A plain white glossy BSDF is 100% reflectance and
  returns the whole cornea at 2.5-4.3 linear -- it whites out both eyes.
* **A mirror shows the source's RADIANCE, which does not fall off with distance.** A
  distant lamp still makes a catchlight, just a small one; the sun makes a tiny brilliant
  spark. Anything that gates on distance gets sunlight exactly backwards.
* **Only the corneas may reach the camera.** Nothing else is visible to camera rays, so
  the pass contains reflections and nothing else and can be ADDED to the photograph.

**The proxy is NOT used as an occluder, and that is a measurement rather than a
preference.** Round 6 recorded that the proxy has no eye sockets; here is what that costs.
With the proxy in the scene as an occluder, frames 20-22 keep only 22-36 percent of the
corneal reflection energy and the peak falls from 2.4-6.1 linear to 0.07-0.57 -- the
cornea is not being shaded by a brow, it is buried inside a closed silhouette shell. The
socket's real job is done by the truncation instead: a cap that stops at the limbus can
only see about 97 degrees around the gaze, which is the front hemisphere the lids leave
open. What is genuinely lost is the brow's own shadow on the upper cornea, and that is
stated rather than hidden.

Eyelid occlusion is likewise not done here. The lid is a 2D matte from the tracked eyelid
polygon and belongs to the compositor; `eye_soft_removal.py --catchlight` applies it.
Asking a lidless proxy for it is what painted silver arcs across the upper lids when
stage 11d was composited.

Writes no production default.
"""
import bpy
import bmesh
from mathutils import Vector
import sys
import json
import math
from pathlib import Path

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--shot', default=None)
ap.add_argument('--start', type=int, default=0)
ap.add_argument('--count', type=int, default=150)
ap.add_argument('--samples', type=int, default=256)
ap.add_argument('--name', default='r34_catchlight')
ap.add_argument('--occluder', action='store_true',
                help='put the proxy head in as an occluder. OFF by default, and measured: '
                     'the proxy is a closed silhouette shell with no eye sockets, so it '
                     'BURIES the cornea rather than shading it. On frames 20-22 it removed '
                     '65-78 percent of the corneal reflection energy and cut the peak by '
                     '8-45x. What the brow really blocks is a fraction of that; what this '
                     'does is delete the catchlight.')
ap.add_argument('--proxy-anchor', action='store_true',
                help='place the cornea where eye_geometry.py put it, on the proxy mesh. '
                     'OFF by default: the proxy and the landmarks disagree about where '
                     'the eye is by up to 3.43 px on a 12 px iris, which walks the '
                     'reflection off the iris onto the sclera.')
ap.add_argument('--raw-anchor', action='store_true',
                help='anchor on the raw landmark instead of a local quadratic fit of its '
                     'own trajectory. A mirror doubles angular error, so the landmark '
                     'measured 0.353 / 0.256 px rms jitter becomes degrees of reflected '
                     'direction and makes small lamps blink.')
ap.add_argument('--hdri-rotation', type=float, default=None,
                help='override the environment rotation, in degrees about Z')
args = ap.parse_args(argv)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
from shot_paths import Shot
from corneal_anchor import smoothed_iris_centres, anchor_centre

shot = Shot(args.shot)
cfg = shot.config
cam_cfg = cfg.get('camera') or {}
LENS = float(cam_cfg.get('lens_mm', 50.0))
SENSOR = float(cam_cfg.get('sensor_width_mm', 36.0))
DIST = float(cam_cfg.get('actor_distance_m', 1.30))
CAM_H = float(cam_cfg.get('height_m', 1.55))
SRC_W, SRC_H = cfg.get('source_resolution', [1920, 1080])
FX = SRC_W * LENS / SENSOR

env_dir = ROOT / (cfg.get('environment') or {}).get('dir', 'scene_library/night_restaurant')
info = json.loads((env_dir / 'scene_info.json').read_text())
hdri = info['hdri']
rotation = args.hdri_rotation if args.hdri_rotation is not None else float(hdri.get('rotation_deg', 0.0))
strength = float((cfg.get('environment') or {}).get('hdri_strength', hdri.get('strength', 1.0)))
HDRI_LIGHT = ROOT / 'assets' / 'hdri' / hdri['lighting']
if not HDRI_LIGHT.exists():
    raise SystemExit('No lighting panorama at %s' % HDRI_LIGHT)

egp = shot.relight / 'eye_geometry.json'
if not egp.exists():
    raise SystemExit('No %s. Run scripts/eye_geometry.py --shot %s first.' % (egp, shot.name))
eye_geo = {r['frame']: r for r in json.loads(egp.read_text())['per_frame']}

proxy_dir = shot.relight / 'proxy'
frames = list(range(args.start, args.start + args.count))


SMOOTH_CENTRES = None if args.raw_anchor else smoothed_iris_centres(eye_geo)


scn = bpy.context.scene
scn.render.engine = 'CYCLES'
scn.cycles.device = 'GPU'
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type != 'CPU'
scn.cycles.samples = args.samples
# Round 1 measured that persistent data serves STALE results and that the error grows
# through a sequence; this moves geometry every frame, which is exactly that case.
scn.render.use_persistent_data = False

# --- world: the shot's own environment, the same graph build_hdri_scene.py writes -------
world = bpy.data.worlds.new('Catchlight environment')
scn.world = world
nt = world.node_tree
nt.nodes.clear()
out = nt.nodes.new('ShaderNodeOutputWorld')
bg = nt.nodes.new('ShaderNodeBackground')
env = nt.nodes.new('ShaderNodeTexEnvironment')
mapping = nt.nodes.new('ShaderNodeMapping')
texco = nt.nodes.new('ShaderNodeTexCoord')
img = bpy.data.images.load(str(HDRI_LIGHT), check_existing=True)
img.colorspace_settings.name = 'Linear Rec.709'
env.image = img
env.projection = 'EQUIRECTANGULAR'
env.interpolation = 'Linear'
nt.links.new(texco.outputs['Generated'], mapping.inputs['Vector'])
nt.links.new(mapping.outputs['Vector'], env.inputs['Vector'])
nt.links.new(env.outputs['Color'], bg.inputs['Color'])
nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
mapping.inputs['Rotation'].default_value[2] = math.radians(rotation)
bg.inputs['Strength'].default_value = strength
world.cycles.sampling_method = 'MANUAL'
# A restaurant interior is a few small bright lamps in a large dark room, and a mirror
# resolves them as individual specks. Under-resolving the light map is how a catchlight
# turns into noise.
world.cycles.sample_map_resolution = 4096
world.cycles.max_bounces = 1024

# --- camera: the plate's own lens and resolution ----------------------------------------
cam_data = bpy.data.cameras.new('Plate camera')
cam_data.lens = LENS
cam_data.sensor_width = SENSOR
cam = bpy.data.objects.new('Plate camera', cam_data)
scn.collection.objects.link(cam)
# Actor stands at the origin; camera looks along +Y toward them, level horizon. Identical
# to build_hdri_scene.py and make_proxy_mesh.py on purpose.
cam.location = (0.0, -DIST, CAM_H)
cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)
scn.camera = cam
scn.render.resolution_x, scn.render.resolution_y = SRC_W, SRC_H
scn.render.resolution_percentage = 100
scn.render.image_settings.media_type = 'IMAGE'
scn.render.image_settings.file_format = 'OPEN_EXR'
scn.render.image_settings.color_mode = 'RGBA'
scn.render.image_settings.color_depth = '16'
scn.render.film_transparent = True
scn.render.filter_size = 1.5

# --- the corneas ------------------------------------------------------------------------
mirror = bpy.data.materials.new('cornea')
mirror.use_nodes = True
mnt = mirror.node_tree
for _n in list(mnt.nodes):
    if _n.type != 'OUTPUT_MATERIAL':
        mnt.nodes.remove(_n)
gl = mnt.nodes.new('ShaderNodeBsdfPrincipled')
gl.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1.0)
gl.inputs['Metallic'].default_value = 0.0
gl.inputs['Roughness'].default_value = 0.02
if 'IOR' in gl.inputs:
    gl.inputs['IOR'].default_value = 1.376
for _n, _v in (('Subsurface Weight', 0.0), ('Sheen Weight', 0.0), ('Coat Weight', 0.0),
               ('Transmission Weight', 0.0), ('Emission Strength', 0.0)):
    if _n in gl.inputs:
        gl.inputs[_n].default_value = _v
mnt.links.new(gl.outputs['BSDF'], next(n for n in mnt.nodes
                                       if n.type == 'OUTPUT_MATERIAL').inputs['Surface'])

# The cornea is a CAP, not a whole sphere, and this is not cosmetic. A full mirror sphere
# has a grazing rim where Fresnel runs to 100%, and that rim sits exactly where the sclera
# and the eyelid are in the photograph. Rendered whole, the left eye's brightest pixel at
# frame 20 was 1.576 linear on the rim, six times the reflection inside the iris, and it
# did not move with the reflected lamp -- it moved with the sphere's silhouette. Truncating
# at the limbus removes it by construction: the real cornea ends there and the sclera
# beyond it is diffuse.
#
#     sin(theta) = iris_radius / cornea_radius = 5.85 / 7.8 = 0.75  ->  theta = 48.6 deg
#
# so the whole cap is inside 48.6 degrees of incidence and Fresnel stays near 2.5-3%.
CAP_COS = math.cos(math.asin(min(0.999, 5.85 / 7.8)))


def corneal_cap(name):
    """A spherical cap of unit radius around local +Z, truncated at the limbus."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=1.0)
    drop = [f for f in bm.faces
            if (sum((v.co.z for v in f.verts)) / len(f.verts)) < CAP_COS]
    bmesh.ops.delete(bm, geom=drop, context='FACES')
    bm.to_mesh(me)
    bm.free()
    if len(me.polygons) == 0:
        raise SystemExit('Corneal cap came out empty; check CAP_COS')
    return me


eye_obs = []
for _side in ('left', 'right'):
    _me = corneal_cap('cornea_' + _side)
    _ob = bpy.data.objects.new('cornea_' + _side, _me)
    scn.collection.objects.link(_ob)
    _ob.data.materials.append(mirror)
    with bpy.context.temp_override(object=_ob, selected_editable_objects=[_ob]):
        bpy.ops.object.shade_smooth()
    eye_obs.append((_side, _ob))

out_dir = shot.composite / args.name
out_dir.mkdir(parents=True, exist_ok=True)

proxy_ob = None
log = []
for fi in frames:
    if proxy_ob is not None:
        me_old = proxy_ob.data
        bpy.data.objects.remove(proxy_ob, do_unlink=True)
        bpy.data.meshes.remove(me_old)
        proxy_ob = None
    if args.occluder:
        mesh_path = proxy_dir / ('%06d.npz' % fi)
        if not mesh_path.exists():
            raise SystemExit('No proxy mesh %s. Run scripts/make_proxy_mesh.py --shot %s.'
                             % (mesh_path, shot.name))
        d = np.load(mesh_path)
        verts, faces = d['verts'].astype(float), d['faces'].astype(int)
        me = bpy.data.meshes.new('proxy')
        me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
        me.validate(verbose=False)
        me.update()
        proxy_ob = bpy.data.objects.new('proxy', me)
        scn.collection.objects.link(proxy_ob)
        # In the scene so the brow, nose and cheek block the environment from reaching the
        # cornea, but never in the picture: this pass is ADDED to the photograph.
        proxy_ob.visible_camera = False

    rec = eye_geo.get(fi) or {}
    rendered = []
    anchor_shift = []
    for _side, _ob in eye_obs:
        e = ((rec.get('eyes') or {}).get(_side)) or {}
        if not e.get('visible'):
            _ob.hide_render = True
            continue
        _ob.hide_render = False
        centre = np.asarray(e['cornea_centre_world'], float)
        radius = float(e['cornea_radius_m'])
        gaze = np.asarray(e.get('gaze_world') or [0.0, -1.0, 0.0], float)
        gaze /= max(np.linalg.norm(gaze), 1e-9)
        shift = 0.0
        if not args.proxy_anchor:
            # Anchor the cornea on the TRACKED IRIS, smoothed, keeping the proxy's depth.
            # Both the why and the arithmetic live in scripts/corneal_anchor.py, because
            # the validator has to place exactly the same sphere.
            px, py = e['iris_centre_px']
            if SMOOTH_CENTRES is not None:
                sm = SMOOTH_CENTRES.get(_side, {}).get(fi)
                if sm is not None:
                    px, py = float(sm[0]), float(sm[1])
            centre, shift = anchor_centre(centre, radius, gaze, (px, py),
                                          (0.0, -DIST, CAM_H), FX, SRC_W, SRC_H)
        _ob.location = tuple(centre)
        _ob.scale = (radius,) * 3
        anchor_shift.append(shift)
        # Aim the cap's local +Z down the measured gaze, so the truncation lands on the
        # limbus rather than somewhere arbitrary on the sphere.
        g = Vector(e['gaze_world'])
        if g.length < 1e-6:
            g = Vector((0.0, -1.0, 0.0))
        _ob.rotation_mode = 'QUATERNION'
        _ob.rotation_quaternion = g.normalized().to_track_quat('Z', 'Y')
        rendered.append(_side)
    scn.render.filepath = str(out_dir / ('%06d.exr' % fi))
    bpy.ops.render.render(write_still=True)
    log.append({'frame': fi, 'eyes_rendered': rendered,
                'landmark_anchor_shift_m': [round(x, 6) for x in anchor_shift]})
    print('R34_CATCH frame', fi, rendered, flush=True)

record = {
    'shot': shot.name,
    'environment': str(env_dir.relative_to(ROOT)),
    'hdri_lighting': hdri['lighting'],
    'hdri_rotation_deg': rotation,
    'hdri_strength': strength,
    'camera': {'lens_mm': LENS, 'sensor_width_mm': SENSOR,
               'location': [0.0, -DIST, CAM_H], 'rotation_euler_deg': [90.0, 0.0, 0.0],
               'resolution': [SRC_W, SRC_H]},
    'cornea': {'bsdf': 'Principled, black base, metallic 0, roughness 0.02, IOR 1.376',
               'why': 'the cornea is a dielectric; Fresnel gives ~2.5% at normal incidence, '
                      'which is why a catchlight is a bright dot and not a mirrored eyeball'},
    'anchor': ('proxy apex (eye_geometry.py)' if args.proxy_anchor
               else ('tracked iris landmark (raw), proxy depth' if args.raw_anchor
                     else 'tracked iris landmark smoothed by a local quadratic (+/-2 '
                          'frames), proxy depth')),
    'occluder': ('proxy mesh, visible_camera False' if args.occluder else None),
    'occluder_note': 'off by default: with the proxy in, frames 20-22 kept only 0.217-0.357 '
                     'of the corneal reflection energy because the shell has no eye sockets',
    'eyelid_occlusion': 'NOT applied here -- the proxy has no lids or sockets; the lid matte '
                        'is 2D, from the tracked eyelid polygon, and belongs to the compositor',
    'samples': args.samples,
    'frames': log,
    'production_enabled': False,
}
(out_dir / 'catchlight.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
shot.log(args.name + '.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print('R34_CATCH_DONE', len(log), 'frames ->', out_dir, flush=True)
