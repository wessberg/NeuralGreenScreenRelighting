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
from mathutils import Vector
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
ap.add_argument('--catchlight', action='store_true',
                help='render ONLY the corneal reflections: a mirror sphere at each eye, the '
                     'plate camera, everything else hidden from camera but still visible to '
                     'reflection rays. What lands in the eye is then whatever the scene actually '
                     'contains -- a bulb 0.5 m away reflects as a bright disc, a window as a soft '
                     'rectangle, an overcast sky as a broad wash, an unlit corner as nothing. '
                     'No threshold decides whether to draw a catchlight; the scene does')
ap.add_argument('--corneal', action='store_true',
                help='round 41: render ONLY the corneal reflections of the tour, with the round '
                     '34 construction -- a dielectric corneal CAP truncated at the limbus, '
                     'anchored on the limbus fitted to the image (--limbus-track), no proxy '
                     'occluder, bulb and fill hidden from camera. Same tour, same exposure '
                     'units as --ratio, so composite_ratio.py --corneal-dir adds it with the '
                     'same anchor it multiplies the plate by')
ap.add_argument('--limbus-track', default='limbus_track_tv_import.json',
                help='--corneal: relight/<track> whose smoothed_centre anchors the cornea. The '
                     'mediapipe iris centre in eye_geometry.json sits 3.08 px from her iris '
                     '(round 35); a mirror doubles angular error')
ap.add_argument('--occluder', action='store_true',
                help='--corneal: keep the proxy head in the scene as an occluder. OFF by '
                     'default and measured (round 34): the shell has no eye sockets and buries '
                     'the cornea, keeping 22-36 percent of the reflection energy')
ap.add_argument('--gaze', choices=('geometry', 'camera'), default='geometry',
                help='--corneal: where the sphere centre sits behind the tracked iris centre. '
                     'geometry = eye_geometry.json gaze (head pose + iris); camera = toward the '
                     'lens. Scored by validate_tour_catchlight.py --plate-glint against the '
                     'plate softbox on 212 open eyes: geometry 1.31 px rms, camera 1.87 with a '
                     '1.3 px lateral bias of opposite sign per eye (round 41)')
ap.add_argument('--raw-anchor', action='store_true',
                help='--corneal: anchor on the per-frame limbus centre instead of its local '
                     'quadratic (+/-2 frames)')
ap.add_argument('--ratio', action='store_true',
                help='render the IRRADIANCE FIELD for the ratio composite instead of a picture '
                     'of her: neutral albedo, linear EXR out. composite_ratio.py turns it into '
                     'a multiplier on the real plate, which is the only path that keeps pores, '
                     'hair strands and eye speculars')
ap.add_argument('--textured', action='store_true',
                help='project the recovered albedo onto the proxy: her real skin, hair, suit '
                     'and microphone, lit by the rig instead of by the plate')
ap.add_argument('--name', default='stress_proxy')
ap.add_argument('--only-output', default='',
                help='comma-separated SOURCE FRAME NUMBERS to render, as named by the mesh files '
                     'and by composite_ratio.py --only-output. The tour still advances across the '
                     'full length, so a selected frame is lit exactly as it would be in a full run')
ap.add_argument('--diagnostic-no-walls', action='store_true', help='control only: opens the proxy volume')
ap.add_argument('--proxy-dir', default='', help='alternate proxy meshes for controlled comparisons')
args = ap.parse_args(argv)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from shot_paths import Shot
from corneal_anchor import smoothed_iris_centres, anchor_limbus

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

proxy_dir = Path(args.proxy_dir) if args.proxy_dir else shot.relight / 'proxy'
frames = sorted(int(p.stem) for p in (shot.relight / 'proxy').glob('*.npz'))
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
# No denoiser on the corneal pass: a bulb reflects in a 7.8 mm cornea as a spot under a
# pixel wide, and a denoiser treats exactly that as noise.
scn.cycles.use_denoising = not args.corneal
# Stepping frames in one process is exactly the case where persistent data serves stale lighting.
scn.render.use_persistent_data = False

world = bpy.data.worlds.new('w')
scn.world = world
wbg = next(n for n in world.node_tree.nodes if n.type == 'BACKGROUND')
wbg.inputs['Strength'].default_value = 0.0

cam_data = bpy.data.cameras.new('cam')
# Demonstration framing: wider than the plate camera so the bulb is visible at the extremes.
# The ratio pass must use the PLATE's own camera, or the multiplier will not line up with the
# photograph it multiplies. The wider demo lens exists only to keep the bulb in frame.
cam_data.lens = LENS if (args.ratio or args.catchlight or args.corneal) else (
    args.demo_lens if args.demo_lens > 0 else LENS)
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

# Ratio pass: a NEUTRAL albedo, so the render IS the irradiance arriving at the surface rather
# than a picture of her. Albedo cancels in E_target/E_source, so putting her real colour here
# would only inject texture noise into a field that should be smooth. Per-channel Jensen radii
# still give the reddish subsurface bleed at a shadow terminator, which is a real lighting cue
# and belongs in the ratio.
RATIO_ALBEDO = 0.6
if args.ratio:
    bsdf.inputs['Base Color'].default_value = (RATIO_ALBEDO,) * 3 + (1.0,)
    # SPECULAR MUST BE OFF IN THE RATIO PASS. Combined = diffuse + subsurface + glossy, and a
    # glossy highlight divided by a diffuse-only E_source is not a lighting ratio: it is a
    # 455x spike. Measured on frame 40 of the tour, E_target went 0.073 at p90 to 19.17 at p99,
    # centred on the top of her head, and the upper clamp turned that into a flat blown patch
    # with a hard edge and no falloff. Specular belongs in the ADDITIVE G term of
    # out = plate x R_d x S + G, never in the multiplier.
    for _n, _v in (('Specular IOR Level', 0.0), ('Metallic', 0.0), ('Coat Weight', 0.0),
                   ('Sheen Weight', 0.0), ('Transmission Weight', 0.0)):
        if _n in bsdf.inputs:
            bsdf.inputs[_n].default_value = _v

albedo_dir = shot.relight / 'albedo'
skin_dir = shot.relight / 'skin_mask'
tex_nodes = None
if args.textured and not args.ratio:
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
# THE BULB MUST NOT BE VISIBLE IN THE RATIO PASS EITHER. It is a demonstration prop: fine in a
# picture of the scene, nonsense inside a multiplier. Rendered into the ratio it contributed a
# 266x187 disc of solid emission -- coverage alpha 0.999, 458x the median irradiance -- which the
# composite then multiplied into the plate as a hard-edged blown patch with no falloff. Anything
# an irradiance pass can SEE ends up scaling the photograph.
if args.ratio or args.corneal:
    bulb_ob.visible_camera = False

# A trace of ambient, so the "key off" stations read as a silhouette against something rather
# than as a black frame. Deliberately far below the rim level.
AMBIENT = 0.012
# Clay albedo is ~0.63/0.44/0.34; the recovered plate albedo averages ~0.16/0.08/0.05 because
# the dark suit and hair dominate it. Same light on both would render the textured version
# roughly three stops down.
EXPOSURE = (args.exposure if args.exposure > 0 else
            (900.0 if args.ratio else 3200.0 if args.textured else 900.0))

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


# The ratio and catchlight passes render at the plate's own resolution, so what they produce is
# pixel-aligned with the photograph they will be combined with.
_linear_pass = args.ratio or args.catchlight or args.corneal
_w = SRC_W if _linear_pass else args.width
scn.render.resolution_x = _w
scn.render.resolution_y = int(round(_w * SRC_H / SRC_W))
scn.render.image_settings.media_type = 'IMAGE'
if _linear_pass:
    # Linear half float: the ratio is a division, and an 8-bit sRGB round trip would quantise
    # the shadow end of it into banding exactly where the shadows are.
    scn.render.image_settings.file_format = 'OPEN_EXR'
    scn.render.image_settings.color_mode = 'RGBA'
    scn.render.image_settings.color_depth = '16'
    scn.render.film_transparent = True
else:
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_mode = 'RGB'
    scn.render.image_settings.color_depth = '8'
if not _linear_pass:
    scn.render.film_transparent = False


def aim(ob, tgt):
    d = np.asarray(tgt, float) - np.asarray(ob.location, float)
    ob.rotation_euler = (math.atan2(math.hypot(d[0], d[1]), d[2]), 0.0,
                         math.atan2(d[1], d[0]) + math.pi / 2)


# --- corneal mirrors -------------------------------------------------------------------------
# A catchlight is a first-surface mirror image of the source, so the honest way to produce one is
# to put a mirror where the cornea is and let Cycles find the source. Everything the artist asked
# to fall out for free does: presence, size, shape, colour and occlusion are all decided by the
# scene rather than by a rule. The one thing worth stating plainly is that a mirror shows the
# source's RADIANCE, which does not fall off with distance -- so a distant lamp still makes a
# catchlight, just a small one. A distance threshold would get sunlight exactly backwards.
eye_geo, eye_obs = None, []
if args.catchlight:
    egp = shot.relight / 'eye_geometry.json'
    if not egp.exists():
        raise SystemExit('No %s. Run scripts/eye_geometry.py --shot %s first.' % (egp, shot.name))
    eye_geo = {r['frame']: r for r in json.loads(egp.read_text())['per_frame']}
    mirror = bpy.data.materials.new('cornea')
    mirror.use_nodes = True
    mnt = mirror.node_tree
    for _n in list(mnt.nodes):
        if _n.type != 'OUTPUT_MATERIAL':
            mnt.nodes.remove(_n)
    # A DIELECTRIC, not a mirror. The first version used a plain glossy BSDF with a white colour,
    # i.e. 100% reflectance, and the whole cornea came back at 2.5-4.3 in linear -- it would have
    # whited out both eyes. The tear film is water on stroma: IOR 1.376, so Fresnel gives about
    # 2.5% at normal incidence rising toward 100% at grazing, which is why a catchlight is a
    # small bright dot rather than a mirrored eyeball, and why the limbus edge picks up a rim.
    # Principled with a BLACK base colour is exactly that lobe and nothing else: the iris
    # diffuse is photographed and comes from the plate, never from here.
    gl = mnt.nodes.new('ShaderNodeBsdfPrincipled')
    gl.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1.0)
    gl.inputs['Metallic'].default_value = 0.0
    # Smooth, not rough: a rough lobe would smear a window into a blob and lose the shape that
    # makes a catchlight readable as a particular kind of source.
    gl.inputs['Roughness'].default_value = 0.02
    if 'IOR' in gl.inputs:
        gl.inputs['IOR'].default_value = 1.376
    for _n, _v in (('Subsurface Weight', 0.0), ('Sheen Weight', 0.0), ('Coat Weight', 0.0),
                   ('Transmission Weight', 0.0), ('Emission Strength', 0.0)):
        if _n in gl.inputs:
            gl.inputs[_n].default_value = _v
    mnt.links.new(gl.outputs['BSDF'], next(n for n in mnt.nodes
                                           if n.type == 'OUTPUT_MATERIAL').inputs['Surface'])
    for _side in ('left', 'right'):
        _me = bpy.data.meshes.new('cornea_' + _side)
        _bm = bmesh.new()
        bmesh.ops.create_uvsphere(_bm, u_segments=48, v_segments=24, radius=1.0)
        _bm.to_mesh(_me)
        _bm.free()
        _ob = bpy.data.objects.new('cornea_' + _side, _me)
        scn.collection.objects.link(_ob)
        _ob.data.materials.append(mirror)
        with bpy.context.temp_override(object=_ob, selected_editable_objects=[_ob]):
            bpy.ops.object.shade_smooth()
        eye_obs.append((_side, _ob))


# --- round 41: the corneal CAP, anchored on the fitted limbus ---------------------------
# Same construction as render_eye_catchlight.py (round 34, validated NCC 0.942 against an
# independent model), lit by THIS tour instead of the shot's panorama. What differs from the
# --catchlight block above, all measured in round 34: the cornea is a cap truncated at the
# limbus (a full sphere has a grazing rim at 100% Fresnel exactly where the lid is); the
# proxy is NOT an occluder (no eye sockets -- it buries the cornea); and the sphere is anchored
# on the limbus fitted to the image, not on the mediapipe landmark.
#     sin(theta) = iris_radius / cornea_radius = 5.85 / 7.8 = 0.75  ->  theta = 48.6 deg
CAP_COS = math.cos(math.asin(min(0.999, 5.85 / 7.8)))
IRIS_R = 0.00585
FX = SRC_W * LENS / SENSOR
corneal_centres, corneal_raw = None, None
if args.corneal:
    egp = shot.relight / 'eye_geometry.json'
    if not egp.exists():
        raise SystemExit('No %s. Run scripts/eye_geometry.py --shot %s first.' % (egp, shot.name))
    eye_geo = {r['frame']: r for r in json.loads(egp.read_text())['per_frame']}
    tp = shot.relight / args.limbus_track
    if not tp.exists():
        raise SystemExit('No limbus track %s' % tp)
    _track = json.loads(tp.read_text(encoding='utf-8'))
    # The anchor module wants eye_geometry-shaped records: {frame: {'eyes': {side: {...}}}}.
    corneal_raw = {}
    for _row in _track['rows']:
        _eyes = {}
        for _e in _row['eyes']:
            if _e.get('fitted'):
                _eyes[_e['side']] = {'visible': True,
                                     'iris_centre_px': list(_e['smoothed_centre']),
                                     'iris_radius_px': float(_e['smoothed_axes'][0])}
        corneal_raw[int(_row['frame'])] = {'eyes': _eyes}
    corneal_centres = None if args.raw_anchor else smoothed_iris_centres(corneal_raw)
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
    for _side in ('left', 'right'):
        _me = bpy.data.meshes.new('cornea_' + _side)
        _bm = bmesh.new()
        bmesh.ops.create_uvsphere(_bm, u_segments=64, v_segments=32, radius=1.0)
        _drop = [f for f in _bm.faces
                 if (sum((v.co.z for v in f.verts)) / len(f.verts)) < CAP_COS]
        bmesh.ops.delete(_bm, geom=_drop, context='FACES')
        _bm.to_mesh(_me)
        _bm.free()
        if len(_me.polygons) == 0:
            raise SystemExit('Corneal cap came out empty; check CAP_COS')
        _ob = bpy.data.objects.new('cornea_' + _side, _me)
        scn.collection.objects.link(_ob)
        _ob.data.materials.append(mirror)
        with bpy.context.temp_override(object=_ob, selected_editable_objects=[_ob]):
            bpy.ops.object.shade_smooth()
        eye_obs.append((_side, _ob))

rig_log = []
catch_log = []
proxy_ob = None
only_output = {int(x) for x in args.only_output.split(',')} if args.only_output else set()
for n, fi in enumerate(frames):
    if only_output and fi not in only_output:
        continue
    t = n / max(len(frames) - 1, 1)                 # 0..1 across the shot
    secs = n / FPS

    if proxy_ob is not None:
        me_old = proxy_ob.data
        bpy.data.objects.remove(proxy_ob, do_unlink=True)
        bpy.data.meshes.remove(me_old)
    mesh_path = proxy_dir / ('%06d.npz' % fi)
    # The timeline comes from the production proxy dir so that an alternate mesh set is lit by the
    # same tour at the same instant. A sparse alternate set therefore only works alongside
    # --only-output, and saying so beats a bare FileNotFoundError 200 frames in.
    if not mesh_path.exists():
        raise SystemExit('No proxy mesh %s. An alternate --proxy-dir that does not cover the whole '
                         'timeline needs --only-output limited to the frames it has.' % mesh_path)
    d = np.load(mesh_path)
    verts, faces = d['verts'].astype(float), d['faces'].astype(int)
    if args.diagnostic_no_walls:
        nf = int(d['n_front'])
        faces = faces[np.all(faces < nf, axis=1) | np.all(faces >= nf, axis=1)]
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
    # The corneal pass keeps the proxy OUT of the scene unless asked (round 34: no eye
    # sockets, it buries the cornea). Its vertices still set the tour's aim point.
    if not args.corneal or args.occluder:
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

    corneal_rec = None
    if args.corneal:
        if args.occluder:
            proxy_ob.visible_camera = False
        for _o in (bulb_ob, fill_ob):
            _o.visible_camera = False
        rec = eye_geo.get(fi) or {}
        corneal_rec = {'out_frame': n, 'shot_frame': fi,
                       'bulb_world': [float(v) for v in bulb_ob.location],
                       'bulb_rgb': [r, g, b], 'bulb_strength': float(strength),
                       'bulb_radius_m': 0.04,
                       'fill_world': [float(v) for v in fill_ob.location],
                       'fill_rgb': [fr, fg_, fb],
                       'fill_strength': float(fill_em.inputs['Strength'].default_value),
                       'eyes': {}}
        for _side, _ob in eye_obs:
            e = ((rec.get('eyes') or {}).get(_side)) or {}
            tr = ((corneal_raw.get(fi) or {}).get('eyes') or {}).get(_side)
            if not e.get('visible') or tr is None:
                _ob.hide_render = True
                corneal_rec['eyes'][_side] = {'rendered': False,
                                              'why': ('no limbus fit' if tr is None
                                                      else 'eye_geometry not visible')}
                continue
            _ob.hide_render = False
            depth_pt = np.asarray(e['cornea_apex_world'], float)
            c_radius = float(e['cornea_radius_m'])
            if args.gaze == 'camera':
                gaze = np.array([0.0, -DIST, CAM_H]) - depth_pt
            else:
                gaze = np.asarray(e.get('gaze_world') or [0.0, -1.0, 0.0], float)
            gaze = gaze / max(float(np.linalg.norm(gaze)), 1e-9)
            px, py = tr['iris_centre_px']
            if corneal_centres is not None:
                sm = corneal_centres.get(_side, {}).get(fi)
                if sm is not None:
                    px, py = float(sm[0]), float(sm[1])
            # The tracked pixel is the IRIS CENTRE, so that is what lands on its ray; the
            # sphere centre sits sqrt(R^2 - r^2) = 5.16 mm behind it along the gaze
            # (corneal_anchor.anchor_limbus). Depth from the proxy, as before.
            centre, iris_w = anchor_limbus(depth_pt, c_radius, IRIS_R, gaze, (px, py),
                                           (0.0, -DIST, CAM_H), FX, SRC_W, SRC_H)
            shift = float(np.linalg.norm(iris_w - depth_pt))
            _ob.location = tuple(centre)
            _ob.scale = (c_radius,) * 3
            gv = (Vector(tuple(gaze)) if float(np.linalg.norm(gaze)) > 1e-6
                  else Vector((0.0, -1.0, 0.0)))
            _ob.rotation_mode = 'QUATERNION'
            _ob.rotation_quaternion = gv.normalized().to_track_quat('Z', 'Y')
            corneal_rec['eyes'][_side] = {'rendered': True,
                                          'centre_world': [float(v) for v in centre],
                                          'radius_m': c_radius,
                                          'gaze_world': [float(v) for v in gaze],
                                          'anchor_px': [float(px), float(py)],
                                          'anchor_shift_m': float(shift),
                                          'iris_radius_px': tr['iris_radius_px']}
        catch_log.append(corneal_rec)
    if args.catchlight:
        # The head stays in the scene as an OCCLUDER -- brow, nose and cheek block part of the
        # environment from reaching the cornea -- but nothing except the mirrors is allowed to
        # reach the camera, so the pass contains only the reflections and can be added.
        proxy_ob.visible_camera = False
        for _o in (bulb_ob, fill_ob):
            _o.visible_camera = False
        rec = eye_geo.get(fi) or {}
        for _side, _ob in eye_obs:
            d = ((rec.get('eyes') or {}).get(_side)) or {}
            if not d.get('visible'):
                _ob.hide_render = True
                continue
            _ob.hide_render = False
            _ob.location = tuple(d['cornea_centre_world'])
            _ob.scale = (d['cornea_radius_m'],) * 3
        catch_log.append({'out_frame': n, 'shot_frame': fi,
                          'eyes_rendered': [k for k, o in eye_obs if not o.hide_render]})

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
          'textured': args.textured, 'ratio': args.ratio, 'catchlight': args.catchlight,
          'corneal': args.corneal,
          'corneal_pass': ({
              'model': 'dielectric corneal CAP truncated at the limbus (sin theta = 5.85/7.8), '
                       'Principled black base, roughness 0.02, IOR 1.376; the round 34 construction',
              'anchor': ('per-frame limbus centre' if args.raw_anchor else
                         'limbus centre smoothed by a local quadratic (+/-2 frames)') +
                        ' from relight/%s, proxy depth; sphere centre 5.16 mm behind the iris '
                        'centre along the %s gaze' % (args.limbus_track, args.gaze),
              'gaze': args.gaze,
              'occluder': 'proxy mesh, camera-invisible' if args.occluder else None,
              'visible_to_camera': 'the corneal caps only; bulb and fill hidden, proxy absent',
              'samples': args.samples, 'denoise': False,
              'cycles_filter': {'type': scn.cycles.pixel_filter_type,
                                'width': scn.cycles.filter_width},
              'per_frame': catch_log} if args.corneal else None),
          'ratio_albedo': RATIO_ALBEDO if args.ratio else None,
          'catchlight_pass': ({
              'model': 'a mirror sphere of the measured corneal radius at each eye, from '
                       'relight/eye_geometry.json, rendered with the PLATE camera',
              'roughness': 0.02, 'ior': 1.376,
              'reflectance': 'dielectric Fresnel, ~2.5% at normal incidence. A 100% mirror '
                             'returned the whole cornea at 2.5-4.3 linear and would have whited '
                             'out both eyes',
              'visible_to_camera': 'the corneal spheres ONLY. The proxy stays in the scene as an '
                                   'occluder -- brow, nose and cheek block part of the '
                                   'environment from reaching the cornea -- but is hidden from '
                                   'camera rays, as are the bulb and the fill, so this pass is '
                                   'purely additive',
              'why_no_threshold': 'presence, size, shape, colour and occlusion all come from the '
                                  'scene. A mirror shows the source RADIANCE, which does not fall '
                                  'off with distance, so a distant lamp still makes a catchlight '
                                  '-- a small one. A distance rule would get sunlight backwards',
              'per_frame': catch_log} if args.catchlight else None),
          'exposure': EXPOSURE,
          'note': ('Albedo projected from the plate: her real skin, hair, suit and microphone, '
                   'with the original studio key divided out and the rig lighting it instead. '
                   'Every colour is a plate pixel scaled by a renderer value.'
                   if args.textured else
                   'Clay-skin proxy only: no plate texture, so every value on screen is computed '
                   'geometry and light rather than photography.'),
          'per_frame': rig_log}
(out_dir / 'render_manifest.json').write_text(json.dumps(report, indent=2))
if not args.only_output:
    # A separate log for the catchlight pass: it shares the tour and therefore the out_frame ->
    # shot_frame mapping, but overwriting stress_proxy.json would leave the ratio composite
    # reading a manifest whose 'ratio' flag says False.
    shot.log('stress_proxy_corneal.json' if args.corneal else
             'stress_proxy_catchlight.json' if args.catchlight
             else 'stress_proxy.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ('frames', 'output_dir', 'rig')}, indent=2))
