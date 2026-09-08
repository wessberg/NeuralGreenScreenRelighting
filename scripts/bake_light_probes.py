"""Bake equirectangular light probes at the actor's position, for the relighter.

    blender -b scene_library/night_restaurant/night_restaurant.blend \
            --python scripts/bake_light_probes.py -- --shot <dir>

Why. relight_controls.py currently starts from a single constant ambient_rgb and an analytic
light list. Under a photographic HDRI there is no light list at all, and a flat ambient throws
away every bit of directional structure plus the room's own bounce -- which in a night interior
is a large fraction of the total. Rendering a small panorama at the actor's position and
projecting it to second-order spherical harmonics captures true incident irradiance including
global illumination, coloured bounce, area-light solid angle and static occlusion, in 27 floats
per probe. A Lambertian BRDF is a low-pass filter that keeps ~99% of its energy in bands 0-2
(Ramamoorthi & Hanrahan 2001), so SH L2 is the correct band-limited representation, and a flat
ambient is just band 0.

Probe positions are derived from the measured face track and the camera model rather than
guessed, and several heights are baked because the vertical gradient across a figure (a lamp
lights head, chest and waist very differently) is itself a strong "composited-in" tell.

Blender 5.x specifics: panorama settings live on camera DATA (camera.data.panorama_type), not
under camera.data.cycles; image_settings.media_type must be set before file_format; the
equirect camera matches the world texture mapping only at object rotation (90, 0, -90).
"""
import bpy, sys, json, math
from pathlib import Path

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--shot', default=None)
ap.add_argument('--size', type=int, default=128, help='probe width; height is half')
ap.add_argument('--samples', type=int, default=512)
ap.add_argument('--frames', type=int, default=0,
                help='bake this many frames of the animation (0 = a single static bake)')
ap.add_argument('--every', type=int, default=1, help='bake every Nth frame; SH interpolates linearly')
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

W_PX, H_PX = cfg.get('source_resolution', [1920, 1080])
fx = W_PX * LENS / SENSOR                      # px; square pixels, horizontal sensor fit
cx, cy = W_PX / 2.0, H_PX / 2.0

# --- probe positions from the measured face track --------------------------------------
track = json.loads((shot.relight / 'face_track.json').read_text())
rows = [r for r in track['per_frame'] if r.get('found')]
mid = rows[len(rows) // 2]
fu, fv = mid['face_centroid_px']
# Back-project the face centroid at the assumed subject distance.
face_x = (fu - cx) * DIST / fx
face_z = CAM_H - (fv - cy) * DIST / fx
# Vertical extent of the visible figure, from the face bbox up and the frame edge down.
head_top_z = CAM_H - (mid['bbox_px'][1] - cy) * DIST / fx
frame_bottom_z = CAM_H - (H_PX - cy) * DIST / fx

probes = [
    {'name': 'head',  'position': [face_x, 0.0, face_z + 0.04]},
    {'name': 'chest', 'position': [face_x, 0.0, face_z - 0.30]},
    {'name': 'waist', 'position': [face_x, 0.0, max(face_z - 0.60, frame_bottom_z + 0.05)]},
]

scn = bpy.context.scene
vl = bpy.context.view_layer

# Anything standing in for the actor must not occlude its own environment.
hidden = []
for ob in bpy.data.objects:
    if ob.name.upper().startswith('PROXY') or ob.name.lower().startswith('actor'):
        hidden.append((ob, ob.hide_render)); ob.hide_render = True

cam_data = bpy.data.cameras.new('SH probe')
cam_data.type = 'PANO'
cam_data.panorama_type = 'EQUIRECTANGULAR'
for attr, val in (('latitude_min', -math.pi / 2), ('latitude_max', math.pi / 2),
                  ('longitude_min', -math.pi), ('longitude_max', math.pi)):
    if hasattr(cam_data, attr):
        setattr(cam_data, attr, val)
cam_data.dof.use_dof = False
probe_cam = bpy.data.objects.new('SH probe', cam_data)
scn.collection.objects.link(probe_cam)
# The rotation that makes an equirect render match the world environment-texture mapping.
probe_cam.rotation_euler = (math.radians(90.0), 0.0, math.radians(-90.0))

previous_cam = scn.camera
scn.camera = probe_cam
scn.render.resolution_x, scn.render.resolution_y = args.size, args.size // 2
scn.render.resolution_percentage = 100
# Persistent data keeps the Cycles scene resident between renders and serves STALE lighting
# when stepping frames in one process: every probe came back byte-identical with it on.
scn.render.use_persistent_data = False
scn.render.use_motion_blur = False
scn.render.film_transparent = False
scn.cycles.samples = args.samples
scn.cycles.use_denoising = False               # denoising would bias the integrated irradiance
scn.cycles.use_adaptive_sampling = False

ims = scn.render.image_settings
if hasattr(ims, 'media_type'):
    ims.media_type = 'IMAGE'
ims.file_format = 'OPEN_EXR'
ims.color_mode = 'RGB'
ims.color_depth = '32'
ims.exr_codec = 'ZIP'
ims.color_management = 'OVERRIDE'
ims.view_settings.view_transform = 'Raw'       # radiance, not a display transform

env_dir = Path(bpy.data.filepath).parent if bpy.data.filepath else (ROOT / 'scene_library/night_restaurant')
probe_dir = env_dir / 'probes'
probe_dir.mkdir(parents=True, exist_ok=True)

frames = list(range(0, args.frames, args.every)) if args.frames else [None]
for f in frames:
    if f is not None:
        scn.frame_set(f)
    for p in probes:
        probe_cam.location = p['position']
        name = p['name'] if f is None else '%s_%06d' % (p['name'], f)
        outfile = probe_dir / ('%s.exr' % name)
        scn.render.filepath = str(outfile)
        bpy.ops.render.render(write_still=True)
        p.setdefault('files', []).append(outfile.name)
    if f is None or f % 25 == 0:
        print('PROBE_FRAME', f, flush=True)

scn.camera = previous_cam
for ob, was in hidden:
    ob.hide_render = was

meta = {
    'scene': env_dir.name,
    'shot': shot.name,
    'convention': ('real SH l<=2, Blender world axes, equirect probe camera at rotation '
                   '(90, 0, -90) which matches the world environment texture mapping'),
    'probe_resolution': [args.size, args.size // 2],
    'samples': args.samples,
    'denoising': False,
    'camera_model': {'lens_mm': LENS, 'sensor_width_mm': SENSOR, 'fx_px': round(fx, 1),
                     'subject_distance_m': DIST, 'camera_height_m': CAM_H},
    'face_centroid_px': [fu, fv],
    'derived_face_world': [round(face_x, 4), 0.0, round(face_z, 4)],
    'head_top_world_z': round(head_top_z, 4),
    'probes': probes,
    'baked_frames': frames if frames != [None] else None,
    'every': args.every,
}
(probe_dir / 'probes.json').write_text(json.dumps(meta, indent=2))
print('PROBES_BAKED', json.dumps({'dir': str(probe_dir), 'probes': [p['name'] for p in probes]}))
