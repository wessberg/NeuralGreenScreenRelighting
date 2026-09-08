"""Hard stress test: a lightbulb flies around the proxy, so self-shadows are visible.

    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" -b --factory-startup \\
        --python scripts/stress_proxy.py -- --shot <dir>

`animate_lights.py` stresses what a *gain map* can track: level, colour, falloff. This stresses
what only geometry can produce -- a shadow that moves.

A **visible lightbulb flies a tour of named stations** around the actor, holding at each long
enough to read and easing between them. Discrete recognisable setups beat a continuous orbit
here: each has a look a gaffer would name, and the shadow it throws is either right or obviously
wrong.

| station | what it proves |
|---|---|
| front, far (2.4 m) | flat frontal base, almost no shadow -- the control |
| front, close (0.5 m) | inverse-square falloff, ~23x |
| below chin | nose shadow thrown **upward** across the eyes |
| above head | nose shadow thrown **downward** onto the lip |
| camera left / right | terminator runs down the nose, one side of the face goes dark |
| behind (backlight) | rim only, face in shadow -- and the bulb disappears behind the head, which is what proves the back shell actually blocks light |
| high three-quarter | the classic key position |

Running continuously on top of the tour: a full **hue** sweep, a beat-locked **strobe** at 120 BPM
with two hard blackouts, and **intensity** changes with inverse-square compensation so the shot
does not simply blow out as the bulb closes in.

The bulb is an emissive MESH, not a light object: Cycles light objects are invisible to camera
rays, which cost round 1 a day, so a light object would throw shadows while staying invisible.
Small (4 cm) on purpose -- a large soft source would wash out the very shadows this exists to
show. Rendered on a **wider demo lens than the plate camera** (35 mm vs 50 mm, same position)
purely so the bulb stays in frame when it swings wide; this is a demonstration framing, not the
shot framing.

Renders the proxy alone as clay-skin. That is the point: with no plate texture, nothing on screen
can be mistaken for photographic detail, so every value is unambiguously computed geometry
and light.
"""
import bpy, bmesh, sys, json, math, argparse, colorsys
from pathlib import Path

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument('--shot', default=None)
ap.add_argument('--frames', type=int, default=0)
ap.add_argument('--start', type=int, default=0)
ap.add_argument('--samples', type=int, default=64)
ap.add_argument('--width', type=int, default=960)
ap.add_argument('--bpm', type=float, default=120.0)
ap.add_argument('--demo-lens', type=float, default=35.0,
                help='wider than the plate lens so the bulb stays in frame; 0 uses the plate lens')
ap.add_argument('--length', type=int, default=360,
                help='output frames; the proxy sequence ping-pongs to fill it')
ap.add_argument('--exposure', type=float, default=0.0,
                help='overall light scale; 0 picks a default per mode. The recovered albedo is '
                     'far darker than the clay stand-in (a dark suit and hair dominate it), so '
                     'the textured render needs several times the level to sit at the same '
                     'exposure')
ap.add_argument('--textured', action='store_true',
                help='project the recovered albedo onto the proxy: her real skin, hair, suit '
                     'and microphone, lit by the rig instead of by the plate')
ap.add_argument('--name', default='stress_proxy')
args = ap.parse_args(argv)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from shot_paths import Shot

shot = Shot(args.shot)
cfg = shot.config
cam_cfg = cfg.get('camera') or {}
LENS = float(cam_cfg.get('lens_mm', 50.0))
SENSOR = float(cam_cfg.get('sensor_width_mm', 36.0))
DIST = float(cam_cfg.get('actor_distance_m', 1.30))
CAM_H = float(cam_cfg.get('height_m', 1.55))
SRC_W, SRC_H = cfg.get('source_resolution', [1920, 1080])
FPS = float(shot.frame_rate)

import numpy as np

proxy_dir = shot.relight / 'proxy'
frames = sorted(int(p.stem) for p in proxy_dir.glob('*.npz'))
if not frames:
    raise SystemExit('No proxy meshes. Run scripts/make_proxy_mesh.py --shot %s first.' % shot.name)
frames = [f for f in frames if f >= args.start]
if args.frames:
    frames = frames[:args.frames]
# The shot is only 150 frames (2.5 s at 59.94). Eleven stations need longer than that to read, so
# the proxy sequence ping-pongs -- forward then backward -- to fill `--length` output frames.
# Ping-pong rather than loop, so the performer's motion stays continuous across the seam.
if args.length and args.length > len(frames):
    cycle = frames + frames[-2:0:-1]
    frames = [cycle[i % len(cycle)] for i in range(args.length)]

out_dir = shot.composite / args.name
out_dir.mkdir(parents=True, exist_ok=True)

# --- scene skeleton, built once ----------------------------------------------------------
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
# Stepping frames in one process is exactly the case where persistent data serves stale lighting.
scn.render.use_persistent_data = False

world = bpy.data.worlds.new('w')
scn.world = world
wbg = next(n for n in world.node_tree.nodes if n.type == 'BACKGROUND')
wbg.inputs['Strength'].default_value = 0.0

cam_data = bpy.data.cameras.new('cam')
# Demonstration framing: wider than the plate camera so the bulb is visible at the extremes.
cam_data.lens = args.demo_lens if args.demo_lens > 0 else LENS
cam_data.sensor_width = SENSOR
cam_data.sensor_fit = 'HORIZONTAL'
cam = bpy.data.objects.new('cam', cam_data)
scn.collection.objects.link(cam)
cam.location = (0.0, -DIST, CAM_H)
cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)
scn.camera = cam

JENSEN_SKIN1_RADIUS_MM = (3.67, 1.37, 0.68)
skin = bpy.data.materials.new('proxy_skin')
skin.use_nodes = True
nt = skin.node_tree
bsdf = nt.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.63, 0.44, 0.34, 1.0)
bsdf.inputs['Roughness'].default_value = 0.45
try:
    bsdf.subsurface_method = 'RANDOM_WALK_SKIN'
except Exception:
    pass
for _name, _val in (('Subsurface Weight', 0.35),
                    ('Subsurface Radius', tuple(r / 1000.0 for r in JENSEN_SKIN1_RADIUS_MM)),
                    ('Subsurface Scale', 1.0)):
    if _name in bsdf.inputs:
        bsdf.inputs[_name].default_value = _val

albedo_dir = shot.relight / 'albedo'
skin_dir = shot.relight / 'skin_mask'
tex_nodes = None
if args.textured:
    if not (albedo_dir / ('%06d.png' % frames[0])).exists():
        raise SystemExit('No albedo. Run scripts/make_albedo.py --shot %s first.' % shot.name)
    # UV-projected albedo, so hair, the suit and the microphone all carry their real colour --
    # not just skin. The UVs are the vertices' own image coordinates, so the projection is exact.
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.interpolation = 'Cubic'
    tex.extension = 'EXTEND'
    nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    # The skin mask gates the skin shader. Subsurface on a wool suit or a foam windshield makes
    # them glow like wax, and rough fabric does not share skin's specular roughness.
    msk = nt.nodes.new('ShaderNodeTexImage')
    msk.interpolation = 'Cubic'
    msk.extension = 'EXTEND'
    msk.image_user.use_auto_refresh = True
    sss = nt.nodes.new('ShaderNodeMath')
    sss.operation = 'MULTIPLY'
    sss.inputs[1].default_value = 0.35
    nt.links.new(msk.outputs['Color'], sss.inputs[0])
    if 'Subsurface Weight' in bsdf.inputs:
        nt.links.new(sss.outputs['Value'], bsdf.inputs['Subsurface Weight'])
    rough = nt.nodes.new('ShaderNodeMapRange')          # skin 0.45, everything else 0.72
    rough.inputs['From Min'].default_value = 0.0
    rough.inputs['From Max'].default_value = 1.0
    rough.inputs['To Min'].default_value = 0.72
    rough.inputs['To Max'].default_value = 0.45
    nt.links.new(msk.outputs['Color'], rough.inputs['Value'])
    nt.links.new(rough.outputs['Result'], bsdf.inputs['Roughness'])
    tex_nodes = (tex, msk)


def emitter(name, size, sphere=False):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    if sphere:
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=size)
    else:
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    scn.collection.objects.link(ob)
    mat = bpy.data.materials.new(name + '_e')
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type != 'OUTPUT_MATERIAL':
            nt.nodes.remove(n)
    em = nt.nodes.new('ShaderNodeEmission')
    nt.links.new(em.outputs['Emission'], nt.nodes['Material Output'].inputs['Surface'])
    ob.data.materials.append(mat)
    return ob, em


# A 4 cm bulb: small enough for a crisp terminator, and visible in frame as a real object.
bulb_ob, bulb_em = emitter('bulb', 0.04, sphere=True)
fill_ob, fill_em = emitter('fill', 0.30)     # 60 cm soft bounce, so the shadow side stays legible
# The fill is a lighting instrument, not set dressing: a 60 cm white panel drifting through frame
# dominates the shot and hides the very thing being demonstrated. Mesh emitters ARE visible to
# camera rays (unlike light objects), so this has to be turned off explicitly.
fill_ob.visible_camera = False

# A trace of ambient, so the "key off" stations read as a silhouette against something rather
# than as a black frame. Deliberately far below the rim level.
AMBIENT = 0.012
# Clay albedo is ~0.63/0.44/0.34; the recovered plate albedo averages ~0.16/0.08/0.05 because
# the dark suit and hair dominate it. Same light on both would render the textured version
# roughly three stops down.
EXPOSURE = args.exposure if args.exposure > 0 else (3200.0 if args.textured else 900.0)

# --- the tour ----------------------------------------------------------------------------
# Offsets are (x right, y away from camera, z up) RELATIVE to the aim point, in metres, so the
# stations follow the performer rather than sitting at fixed world coordinates.
# (name, offset, fill_scale). fill_scale 0 kills the front bounce entirely, which is how the
# "key off, only a source behind" station becomes a true near-silhouette rather than a lit face
# with a rim added on top.
STATIONS = [
    ('front far',            (0.00, -2.40, 0.10), 1.0),
    ('front close',          (0.00, -0.50, 0.05), 1.0),
    ('below chin',           (0.00, -0.45, -0.55), 0.6),
    ('above head',           (0.00, -0.35, 0.75), 0.6),
    ('camera left',          (-0.80, -0.30, 0.10), 0.5),
    ('camera right',         (0.80, -0.30, 0.10), 0.5),
    ('behind rim',           (0.55, 0.85, 0.45), 0.5),
    # Well off-axis on purpose. Directly behind her the bulb is occluded by her own body and the
    # frame goes completely black -- physically correct and useless as a demonstration.
    ('KEY OFF: back only',   (0.75, 0.80, 0.55), 0.0),
    ('KEY OFF: back only',   (-0.75, 0.80, 0.35), 0.0),
    ('high three-quarter',   (-0.60, -0.60, 0.70), 0.8),
    ('front far',            (0.00, -2.40, 0.10), 1.0),
]
HOLD = 0.45                       # fraction of each segment spent parked at the station


def station_at(t):
    """Ease between stations, holding at each so every setup is readable."""
    segs = len(STATIONS) - 1
    u = min(max(t, 0.0), 0.999999) * segs
    i = int(u)
    f = u - i
    f = 0.0 if f < HOLD else (lambda x: x * x * (3 - 2 * x))((f - HOLD) / (1 - HOLD))
    a, b = np.array(STATIONS[i][1]), np.array(STATIONS[i + 1][1])
    fa, fb = STATIONS[i][2], STATIONS[i + 1][2]
    return (a + (b - a) * f, (STATIONS[i][0] if f < 0.5 else STATIONS[i + 1][0]),
            fa + (fb - fa) * f)


scn.render.resolution_x = args.width
scn.render.resolution_y = int(round(args.width * SRC_H / SRC_W))
scn.render.image_settings.media_type = 'IMAGE'
scn.render.image_settings.file_format = 'PNG'
scn.render.image_settings.color_mode = 'RGB'
scn.render.image_settings.color_depth = '8'
scn.render.film_transparent = False


def aim(ob, tgt):
    d = np.asarray(tgt, float) - np.asarray(ob.location, float)
    ob.rotation_euler = (math.atan2(math.hypot(d[0], d[1]), d[2]), 0.0,
                         math.atan2(d[1], d[0]) + math.pi / 2)


rig_log = []
proxy_ob = None
for n, fi in enumerate(frames):
    t = n / max(len(frames) - 1, 1)                 # 0..1 across the shot
    secs = n / FPS

    if proxy_ob is not None:
        me_old = proxy_ob.data
        bpy.data.objects.remove(proxy_ob, do_unlink=True)
        bpy.data.meshes.remove(me_old)
    d = np.load(proxy_dir / ('%06d.npz' % fi))
    verts, faces = d['verts'].astype(float), d['faces'].astype(int)
    me = bpy.data.meshes.new('proxy')
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.validate(verbose=False)
    me.update()
    if args.textured:
        if 'uv' not in d:
            raise SystemExit('Proxy %06d has no UVs. Rebuild with scripts/make_proxy_mesh.py.' % fi)
        uv = d['uv'].astype(float)
        layer = me.uv_layers.new(name='plate')
        # Per-loop, not per-vertex: a loop is one corner of one face, so the UV is looked up
        # through the loop's vertex index.
        flat = np.empty((len(me.loops), 2), float)
        for li, loop in enumerate(me.loops):
            flat[li] = uv[loop.vertex_index]
        layer.data.foreach_set('uv', flat.ravel())
        me.update()
    proxy_ob = bpy.data.objects.new('proxy', me)
    scn.collection.objects.link(proxy_ob)
    proxy_ob.data.materials.append(skin)
    with bpy.context.temp_override(object=proxy_ob, selected_editable_objects=[proxy_ob]):
        bpy.ops.object.shade_smooth()

    zs, xs = verts[:, 2], verts[:, 0]
    head_z, mid_z = float(zs.max()), float(np.percentile(zs, 60))
    target = (float(np.median(xs)), 0.0, (head_z + mid_z) / 2)

    if tex_nodes is not None:
        for node, folder in zip(tex_nodes, (albedo_dir, skin_dir)):
            if node.image is not None:
                bpy.data.images.remove(node.image)
            img = bpy.data.images.load(str(folder / ('%06d.png' % fi)))
            # Albedo is authored in sRGB; the mask is data and must not be transformed.
            img.colorspace_settings.name = 'sRGB' if folder is albedo_dir else 'Non-Color'
            node.image = img

    offset, station, fill_scale = station_at(t)
    bulb_ob.location = (target[0] + offset[0], target[1] + offset[1], target[2] + offset[2])
    aim(bulb_ob, target)
    radius = float(np.linalg.norm(offset))

    hue = t % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 1.0)
    # Beat-locked strobe with hard blackouts, and inverse-square compensation so the shot does
    # not simply blow out as the bulb closes in.
    beat = (secs * args.bpm / 60.0) % 1.0
    pulse = 0.25 + 0.75 * math.exp(-6.0 * beat)
    blackout = 0.06 if (0.62 < t < 0.645 or 0.855 < t < 0.875) else 1.0
    # Inverse-square compensation keeps the exposure sane as the bulb closes in. Backlight needs
    # more level than a front key: only a grazing sliver of the silhouette returns any light.
    back = max(0.0, min(1.0, (offset[1] + 0.2) / 0.8))       # 0 in front, 1 well behind
    strength = EXPOSURE * (1.0 + 3.0 * back) * pulse * blackout * (radius / 2.40) ** 2
    bulb_em.inputs['Strength'].default_value = strength
    # Lift the ambient floor only where the key is off, so she stays a readable silhouette.
    wbg.inputs['Strength'].default_value = AMBIENT * (1.0 - fill_scale) * blackout
    bulb_em.inputs['Color'].default_value = (r, g, b, 1.0)

    fr, fg_, fb = colorsys.hsv_to_rgb((hue + 0.5) % 1.0, 0.45, 1.0)
    fill_ob.location = (target[0] - offset[0] * 0.6, target[1] - offset[1] * 0.6, target[2] - 0.2)
    aim(fill_ob, target)
    fill_em.inputs['Strength'].default_value = 4.0 * (EXPOSURE / 900.0) * blackout * fill_scale
    fill_em.inputs['Color'].default_value = (fr, fg_, fb, 1.0)

    scn.render.filepath = str(out_dir / ('%06d' % n))
    bpy.ops.render.render(write_still=True)
    rig_log.append({'out_frame': n, 'shot_frame': fi, 'station': station,
                    'bulb_offset_m': [round(float(v), 3) for v in offset],
                    'bulb_distance_m': round(radius, 3),
                    'bulb_strength': round(strength, 1),
                    'hue': round(hue, 3), 'blackout': blackout != 1.0,
                    'fill_scale': round(float(fill_scale), 3)})
    if n % 10 == 0:
        print('PROXY %d/%d  %-20s d %.2f m  hue %.2f%s'
              % (n, len(frames), station, radius, hue,
                 '  BLACKOUT' if blackout != 1.0 else ''), flush=True)

report = {'shot': shot.name, 'frames': len(frames), 'fps': str(shot.frame_rate),
          'output_dir': str(out_dir), 'samples': args.samples,
          'rig': {'stations': [s[0] for s in STATIONS],
                  'distance_m': [0.50, 2.40],
                  'inverse_square_ratio': round((2.40 / 0.50) ** 2, 1),
                  'hue_sweep': 'full wheel', 'bpm': args.bpm, 'blackouts': 2,
                  'bulb_radius_m': 0.04, 'bulb_visible_to_camera': True, 'exposure': EXPOSURE,
                  'fill_size_m': 0.60, 'demo_lens_mm': args.demo_lens,
                  'plate_lens_mm': LENS,
                  'key_off_stations': 'fill_scale 0: no front light at all, only a source behind '
                                      'the actor -- near-silhouette with a rim'},
          'textured': args.textured,
          'note': ('Albedo projected from the plate: her real skin, hair, suit and microphone, '
                   'with the original studio key divided out and the rig lighting it instead. '
                   'Every colour is a plate pixel scaled by a renderer value.'
                   if args.textured else
                   'Clay-skin proxy only: no plate texture, so every value on screen is computed '
                   'geometry and light rather than photography.'),
          'per_frame': rig_log}
shot.log('stress_proxy.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ('frames', 'output_dir', 'rig')}, indent=2))
