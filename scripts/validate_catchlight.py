"""Does the catchlight move the way the light moves? One source, four positions, measured.

    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" -b --factory-startup \\
        --python scripts/validate_catchlight.py -- --shot <dir>

The companion to ``validate_proxy_shadows.py``, and it exists for the same reason: a sign or
handedness error in this domain is invisible in a single frame and ungradeable afterwards. A
catchlight in the wrong place still looks like a catchlight.

**It has to be a dedicated render, not a measurement off the tour.** The first attempt used the
stress tour and got corr(bulb dx, glint dx) = -0.192 over 93 frames -- apparently a mirrored
axis. It is not: ``stress_proxy.py`` has a fill panel at ``target - offset*0.6``, i.e. on the
OPPOSITE side from the bulb, and the fill is a 0.60 m source against a 0.04 m bulb. In a mirror
the brightest point tracks whichever source wins, so the metric was measuring the fill half the
time. Restricted to frames with the fill up it comes out +0.243 and -0.254, both correct. That is
suggestive rather than conclusive, which is why this exists: one source, moved on purpose.

The test is the movement, not the position. An absolute position depends on gaze, on the corneal
radius and on where the proxy put the eye; the DIRECTION it travels when the light moves depends
only on the reflection geometry, so that is what is asserted:

    light UP     -> glint moves UP    in the image (image y is DOWN, so dy negative)
    light DOWN   -> glint moves DOWN
    light LEFT   -> glint moves LEFT
    light RIGHT  -> glint moves RIGHT

and the magnitude is checked against the half-vector prediction, because the glint moves by
roughly HALF the angle the light moves and a model that got that factor wrong would still pass a
pure direction test.
"""
import bpy, bmesh, sys, json, math
from pathlib import Path

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--shot', default=None)
ap.add_argument('--frame', type=int, default=21)
ap.add_argument('--samples', type=int, default=256)
ap.add_argument('--throw', type=float, default=1.0, help='metres the source is offset')
ap.add_argument('--distance', type=float, default=1.2, help='metres in front of the eye')
ap.add_argument('--tolerance', type=float, default=0.35,
                help='allowed relative error on the predicted travel')
args = ap.parse_args(argv)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
from shot_paths import Shot

shot = Shot(args.shot)
cfg = shot.config
cam_cfg = cfg.get('camera') or {}
LENS = float(cam_cfg.get('lens_mm', 50.0))
SENSOR = float(cam_cfg.get('sensor_width_mm', 36.0))
DIST = float(cam_cfg.get('actor_distance_m', 1.30))
CAM_H = float(cam_cfg.get('height_m', 1.55))
SRC_W, SRC_H = cfg.get('source_resolution', [1920, 1080])

eg = json.loads((shot.relight / 'eye_geometry.json').read_text())
rec = next((r for r in eg['per_frame'] if r['frame'] == args.frame), None)
eyes = {k: v for k, v in ((rec or {}).get('eyes') or {}).items() if v.get('visible')}
if not eyes:
    raise SystemExit('No visible eye on frame %d. Run eye_geometry.py, or pick another frame.'
                     % args.frame)

for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
scn = bpy.context.scene
scn.render.engine = 'CYCLES'
scn.cycles.device = 'GPU'
try:
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type != 'CPU')
except Exception as e:
    print('GPU setup skipped: %s' % e)
scn.cycles.samples = args.samples
scn.cycles.use_denoising = True
scn.render.use_persistent_data = False          # serves stale lighting; see AGENTS.md
world = bpy.data.worlds.new('w')
scn.world = world
next(n for n in world.node_tree.nodes if n.type == 'BACKGROUND').inputs['Strength'].default_value = 0.0

cam_data = bpy.data.cameras.new('cam')
cam_data.lens, cam_data.sensor_width = LENS, SENSOR
cam_data.sensor_fit = 'HORIZONTAL'
cam = bpy.data.objects.new('cam', cam_data)
scn.collection.objects.link(cam)
cam.location = (0.0, -DIST, CAM_H)
cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)
scn.camera = cam
scn.render.resolution_x, scn.render.resolution_y = SRC_W, SRC_H
scn.render.image_settings.media_type = 'IMAGE'
scn.render.image_settings.file_format = 'OPEN_EXR'
scn.render.image_settings.color_mode = 'RGBA'
scn.render.image_settings.color_depth = '16'
scn.render.film_transparent = True

# Corneal mirrors: identical material to stress_proxy.py --catchlight, so this validates the
# thing that ships rather than a lookalike.
mirror = bpy.data.materials.new('cornea')
mirror.use_nodes = True
mnt = mirror.node_tree
for n in list(mnt.nodes):
    if n.type != 'OUTPUT_MATERIAL':
        mnt.nodes.remove(n)
gl = mnt.nodes.new('ShaderNodeBsdfPrincipled')
gl.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1.0)
gl.inputs['Metallic'].default_value = 0.0
gl.inputs['Roughness'].default_value = 0.02
if 'IOR' in gl.inputs:
    gl.inputs['IOR'].default_value = 1.376
mnt.links.new(gl.outputs['BSDF'],
              next(n for n in mnt.nodes if n.type == 'OUTPUT_MATERIAL').inputs['Surface'])

eye_obs = {}
for side, d in eyes.items():
    me = bpy.data.meshes.new('c_' + side)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=1.0)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new('c_' + side, me)
    scn.collection.objects.link(ob)
    ob.data.materials.append(mirror)
    ob.location = tuple(d['cornea_centre_world'])
    ob.scale = (d['cornea_radius_m'],) * 3
    with bpy.context.temp_override(object=ob, selected_editable_objects=[ob]):
        bpy.ops.object.shade_smooth()
    eye_obs[side] = ob

# A single small emissive quad, invisible to camera. One source, so nothing competes for
# "brightest point in the iris" -- which is exactly what went wrong measuring off the tour.
me = bpy.data.meshes.new('src')
bm = bmesh.new()
bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.05)
bm.to_mesh(me)
bm.free()
src = bpy.data.objects.new('src', me)
scn.collection.objects.link(src)
mat = bpy.data.materials.new('src_e')
mat.use_nodes = True
snt = mat.node_tree
for n in list(snt.nodes):
    if n.type != 'OUTPUT_MATERIAL':
        snt.nodes.remove(n)
em = snt.nodes.new('ShaderNodeEmission')
em.inputs['Strength'].default_value = 2000.0
snt.links.new(em.outputs['Emission'],
              next(n for n in snt.nodes if n.type == 'OUTPUT_MATERIAL').inputs['Surface'])
src.data.materials.append(mat)
src.visible_camera = False

anchor = np.array(list(eyes.values())[0]['cornea_apex_world'], float)
fx = SRC_W * LENS / SENSOR
out_dir = shot.composite / 'catchlight_validation'
out_dir.mkdir(parents=True, exist_ok=True)


def project(P):
    Zc = P[1] + DIST
    return np.array([P[0] * fx / Zc + SRC_W / 2 - 0.5,
                     (CAM_H - P[2]) * fx / Zc + SRC_H / 2 - 0.5])


def predicted(side, lpos):
    """Where the half-vector puts the glint, in iris radii from the corneal apex."""
    d = eyes[side]
    C = np.array(d['cornea_centre_world'], float)
    A = np.array(d['cornea_apex_world'], float)
    l = lpos - A
    l /= np.linalg.norm(l)
    v = np.array([0.0, -DIST, CAM_H]) - A
    v /= np.linalg.norm(v)
    hv = l + v
    hv /= np.linalg.norm(hv)
    p = C + hv * d['cornea_radius_m']
    return (project(p) - project(A)) / d['iris_radius_px_desqueezed']


def read_exr(path):
    img = bpy.data.images.load(str(path))
    w, h = img.size
    buf = np.array(img.pixels[:], np.float32).reshape(h, w, 4)[::-1]
    bpy.data.images.remove(img)
    return buf


CASES = {'centre': (0.0, 0.0), 'up': (0.0, +args.throw), 'down': (0.0, -args.throw),
         'left': (-args.throw, 0.0), 'right': (+args.throw, 0.0)}
results = {}
for name, (dx, dz) in CASES.items():
    src.location = (anchor[0] + dx, anchor[1] - args.distance, anchor[2] + dz)
    scn.render.filepath = str(out_dir / name)
    bpy.ops.render.render(write_still=True)
    buf = read_exr(out_dir / (name + '.exr'))
    L = buf[..., 0] * 0.2126 + buf[..., 1] * 0.7152 + buf[..., 2] * 0.0722
    per = {}
    for side, d in eyes.items():
        cx, cy = d['iris_centre_px']
        r = d['iris_radius_px_desqueezed']
        yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]]
        disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * 1.15) ** 2
        if disc.sum() < 20 or L[disc].max() <= 0:
            continue
        m = disc & (L >= 0.5 * L[disc].max())
        wy, wx = np.nonzero(m)
        ww = L[m]
        gx, gy = float((wx * ww).sum() / ww.sum()), float((wy * ww).sum() / ww.sum())
        A = project(np.array(d['cornea_apex_world'], float))
        per[side] = {'measured': [(gx - A[0]) / r, (gy - A[1]) / r],
                     'predicted': predicted(side, np.array(src.location, float)).tolist(),
                     'peak': float(L[disc].max()), 'area_px': int(m.sum())}
    results[name] = per
    print('%-7s source at %s  ->  %s' % (name, tuple(round(v, 3) for v in src.location),
                                         {k: [round(x, 3) for x in v['measured']]
                                          for k, v in per.items()}), flush=True)

checks, ok_all = [], True
for axis, hi, lo, comp, want, why in (
        ('vertical', 'up', 'down', 1, -1, 'image y is DOWN, so a light moving UP must move the '
                                          'glint to a SMALLER y'),
        ('horizontal', 'right', 'left', 0, +1, 'a light moving to +x must move the glint to +x')):
    for side in eyes:
        if side not in results[hi] or side not in results[lo]:
            continue
        m = results[hi][side]['measured'][comp] - results[lo][side]['measured'][comp]
        p = results[hi][side]['predicted'][comp] - results[lo][side]['predicted'][comp]
        sign_ok = (m * want) > 0
        mag_err = abs(m - p) / max(abs(p), 1e-6)
        ok = bool(sign_ok and mag_err <= args.tolerance)
        ok_all = ok_all and ok
        checks.append({'axis': axis, 'eye': side, 'travel_measured_iris_radii': round(m, 4),
                       'travel_predicted_iris_radii': round(p, 4),
                       'relative_error': round(mag_err, 4),
                       'direction_correct': bool(sign_ok), 'pass': ok, 'why': why})
        print('%-10s %-6s travel measured %+.3f  predicted %+.3f  rel err %.2f  %s'
              % (axis, side, m, p, mag_err, 'PASS' if ok else 'FAIL'), flush=True)

report = {'shot': shot.name, 'frame': args.frame, 'samples': args.samples,
          'throw_m': args.throw, 'source_distance_m': args.distance,
          'tolerance': args.tolerance,
          'model': 'one 10 cm emissive quad, invisible to camera, moved between five positions; '
                   'dielectric corneal spheres identical to stress_proxy.py --catchlight',
          'why_dedicated': 'measured off the stress tour instead, corr(bulb dx, glint dx) came '
                           'out -0.192 over 93 frames, which looks like a mirrored axis and is '
                           'not: the tour has a fill panel at target - offset*0.6, on the '
                           'OPPOSITE side from the bulb and 15x its size, so in a mirror the '
                           'brightest point tracked the fill. With the fill up it reads +0.243 '
                           'and -0.254, both correct',
          'what_is_asserted': 'the DIRECTION the glint travels when the source moves, and the '
                              'magnitude against the half-vector prediction -- the glint moves '
                              'by roughly HALF the angle the light moves, and a model that got '
                              'that factor wrong would still pass a pure direction test',
          'checks': checks, 'pass': ok_all, 'per_position': results}
shot.log('catchlight_validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps({'pass': ok_all, 'checks': checks}, indent=2))
