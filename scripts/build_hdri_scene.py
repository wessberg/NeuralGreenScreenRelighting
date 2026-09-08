"""Build a photographic HDRI environment instead of a modelled room.

Run under Blender, not the pipeline venv:

    blender -b --factory-startup --python scripts/build_hdri_scene.py -- --shot <dir> [--test]

Rationale. The modelled listening room reads as CG because its SHAPES are procedural
primitives, which no amount of texturing fixes. A real unclipped panorama is a photograph, so
it cannot look like CG; and because it lights the scene as well as filling the frame, the
actor's illumination and the background come from the SAME photograph and cannot disagree.
That agreement is what "I was actually there" rests on.

Resolution is not the usual objection here. At 50 mm on a 36 mm sensor the horizontal FOV is
39.6 degrees, so a 16K equirect shows 16384*39.6/360 = 1802 px across the frame. The plate is
1080p upscaled 2x, carrying ~1920 px of real detail, so the panorama is at 94% of it.

Two panorama copies are used via a Light Path split: camera rays see the 16K copy, every
lighting ray samples a 4K copy. That keeps the importance map small and the sampling quiet
while the visible backdrop stays sharp.

Blender 5.x API notes honoured here: bpy.data.worlds.new() creates its node tree (use_nodes is
a deprecated no-op); image_settings.media_type must be set before file_format; string socket
keys match identifier-first and skip unavailable sockets; cycles.denoising_use_gpu defaults to
False; panorama settings live on the camera data, not camera.data.cycles.
"""
import bpy, sys, json, math, os
from pathlib import Path

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--shot', default=None)
ap.add_argument('--test', action='store_true')
ap.add_argument('--hdri-rotation', type=float, default=0.0, help='degrees about Z')
args = ap.parse_args(argv)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from shot_paths import Shot                      # stdlib-only

shot = Shot(args.shot)
cfg = shot.config
ENV_DIR = ROOT / 'scene_library' / 'night_restaurant'
ENV_DIR.mkdir(parents=True, exist_ok=True)
HDRI_LIGHT = ROOT / 'assets/hdri/warm_restaurant_night_4k.exr'
HDRI_VIEW = ROOT / 'assets/hdri/warm_restaurant_night_16k.hdr'
for f in (HDRI_LIGHT, HDRI_VIEW):
    if not f.exists():
        raise SystemExit('Missing HDRI: %s' % f)

# --- camera match ---------------------------------------------------------------------
cam_cfg = cfg.get('camera') or {}
LENS = float(cam_cfg.get('lens_mm', 50.0))
SENSOR = float(cam_cfg.get('sensor_width_mm', 36.0))
SUBJECT_DISTANCE = float(cam_cfg.get('actor_distance_m', 1.30))
CAM_HEIGHT = float(cam_cfg.get('height_m', 1.55))

scn = bpy.context.scene
vl = bpy.context.view_layer

# Working space is no longer guaranteed to be Linear Rec.709 in 5.x, and the project's numpy
# stage hard-codes Rec.709 -> Rec.2020 matrices. Assert rather than assume.
ws = getattr(bpy.data, 'colorspace', None)
working = getattr(ws, 'working_space', None) if ws else None
if working is not None and 'Rec.709' not in str(working):
    raise SystemExit('Working space is %r; the numpy pipeline assumes Linear Rec.709' % working)

for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)

scn.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
try:
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type in {'OPTIX', 'CUDA'})
    scn.cycles.device = 'GPU'
except Exception as e:
    print('GPU setup fell back to CPU:', e)

# --- world: two panorama copies, split by Light Path ----------------------------------
world = bpy.data.worlds.new('Night restaurant')
scn.world = world
nt = world.node_tree
nt.nodes.clear()
out = nt.nodes.new('ShaderNodeOutputWorld')
bg = nt.nodes.new('ShaderNodeBackground')
mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'
lp = nt.nodes.new('ShaderNodeLightPath')
env_light = nt.nodes.new('ShaderNodeTexEnvironment')
env_view = nt.nodes.new('ShaderNodeTexEnvironment')
mapping = nt.nodes.new('ShaderNodeMapping')
texco = nt.nodes.new('ShaderNodeTexCoord')

img_light = bpy.data.images.load(str(HDRI_LIGHT), check_existing=True)
img_view = bpy.data.images.load(str(HDRI_VIEW), check_existing=True)
for img in (img_light, img_view):
    # EXR/HDR declare the default-float role, which resolves to Linear Rec.709 automatically.
    img.colorspace_settings.name = 'Linear Rec.709'
env_light.image, env_view.image = img_light, img_view
for e in (env_light, env_view):
    e.projection = 'EQUIRECTANGULAR'
    e.interpolation = 'Linear'

nt.links.new(texco.outputs['Generated'], mapping.inputs['Vector'])
nt.links.new(mapping.outputs['Vector'], env_light.inputs['Vector'])
nt.links.new(mapping.outputs['Vector'], env_view.inputs['Vector'])
nt.links.new(lp.outputs['Is Camera Ray'], mix.inputs['Factor'])
nt.links.new(env_light.outputs['Color'], mix.inputs['A'])   # every lighting ray
nt.links.new(env_view.outputs['Color'], mix.inputs['B'])    # camera rays only
nt.links.new(mix.outputs['Result'], bg.inputs['Color'])
nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
mapping.inputs['Rotation'].default_value[2] = math.radians(args.hdri_rotation)

# Exposure anchor: the plate's performer sits at scene-linear ~0.164 median. Keep the room's
# diffuse mid-tones slightly under that so the figure separates from the ground.
bg.inputs['Strength'].default_value = float(cfg.get('environment', {}).get('hdri_strength', 1.0))

world.cycles.sampling_method = 'MANUAL'
world.cycles.sample_map_resolution = 4096   # a small bright window in a 16K interior needs this
world.cycles.max_bounces = 1024

# --- shadow catcher floor -------------------------------------------------------------
bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0.0, 0.0, 0.0))
floor = bpy.context.active_object
floor.name = 'Shadow catcher floor'
floor.is_shadow_catcher = True
mat = bpy.data.materials.new('Floor catcher')
bsdf = mat.node_tree.nodes.get('Principled BSDF')
if bsdf:
    bsdf.inputs['Base Color'].default_value = (0.18, 0.16, 0.15, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.55
floor.data.materials.append(mat)

# --- camera ---------------------------------------------------------------------------
cam_data = bpy.data.cameras.new('Plate camera')
cam_data.type = 'PERSP'
cam_data.lens = LENS
cam_data.sensor_width = SENSOR
cam_data.sensor_fit = 'HORIZONTAL'
cam_data.dof.use_dof = True
cam_data.dof.focus_distance = SUBJECT_DISTANCE
# A dim interior is never shot at f/5.6. A fast prime near wide open is both plausible and
# one of the most immediate "this was photographed" cues.
cam_data.dof.aperture_fstop = float(cam_cfg.get('aperture_fstop', 2.0))
cam_data.dof.aperture_blades = 0

# --- anamorphic ---------------------------------------------------------------------
# Opt-in per shot, because the artist shoots anamorphic and the stock reporter plate does not.
# All the 1.6x arithmetic lives in scripts/anamorphic.py so no factor of 1.6 is ever retyped.
ANAMORPHIC = bool(cam_cfg.get('anamorphic', False))
anamorphic_report = None
if ANAMORPHIC:
    import anamorphic as ana
    m = ana.model(focal_mm=float(cam_cfg.get('taking_focal_mm', ana.FOCAL_MM)),
                  squeeze=float(cam_cfg.get('squeeze_factor', ana.SQUEEZE)),
                  delivery=tuple(shot.resolution),
                  taking_fstop=float(cam_cfg.get('aperture_fstop', 2.9)))
    # 1.6 is the measured value for a vertical oval; see logs/anamorphic_validation.json.
    ana.apply_to_camera(cam_data, scn, m,
                        aperture_ratio=float(cam_cfg.get('squeeze_factor', ana.SQUEEZE)),
                        focus_distance=SUBJECT_DISTANCE)
    LENS, SENSOR = m['render_focal_mm'], m['render_sensor_width_mm']
    anamorphic_report = m
    print('anamorphic: rendering %dx%d at %.3f mm equivalent, hFOV %.2f deg, f/%.3f, bokeh %s'
          % (m['render_resolution'][0], m['render_resolution'][1], m['render_focal_mm'],
             m['hfov_deg'], m['render_fstop'], m['bokeh']))

cam = bpy.data.objects.new('Plate camera', cam_data)
scn.collection.objects.link(cam)
# Actor stands at the origin; camera looks along +Y toward them, level horizon.
cam.location = (0.0, -SUBJECT_DISTANCE, CAM_HEIGHT)
cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)
scn.camera = cam

# --- render settings ------------------------------------------------------------------
if args.test:
    res = (960, 540)
elif ANAMORPHIC:
    res = tuple(anamorphic_report['render_resolution'])
else:
    res = tuple(shot.resolution)
scn.render.resolution_x, scn.render.resolution_y = res
scn.render.resolution_percentage = 100
scn.render.film_transparent = False          # composite.py composites over an opaque background
scn.cycles.samples = 128 if args.test else 1024
scn.cycles.adaptive_threshold = 0.005
scn.cycles.adaptive_min_samples = 64
scn.cycles.use_denoising = True
scn.cycles.denoiser = 'OPENIMAGEDENOISE'
scn.cycles.denoising_prefilter = 'ACCURATE'
scn.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
scn.cycles.denoising_use_gpu = True          # defaults to False; flip explicitly
scn.render.use_persistent_data = True
vl.cycles.use_pass_shadow_catcher = True

ims = scn.render.image_settings
if hasattr(ims, 'media_type'):
    ims.media_type = 'IMAGE'                 # must precede file_format in 5.x
ims.file_format = 'OPEN_EXR'
ims.color_mode = 'RGBA'
ims.color_depth = '32'
ims.exr_codec = 'ZIP'                        # lossless; never DWAA/DWAB/PXR24
ims.color_management = 'FOLLOW_SCENE'        # writes scene-linear, does not bake the view transform

hfov = 2 * math.degrees(math.atan(SENSOR / (2 * LENS)))
info = {
    'scene': 'night_restaurant',
    'shot': shot.name,
    'hdri': {'lighting': HDRI_LIGHT.name, 'visible': HDRI_VIEW.name,
             'rotation_deg': args.hdri_rotation,
             'strength': bg.inputs['Strength'].default_value,
             'licence': 'CC0, Poly Haven, Greg Zaal / Jarod Guest'},
    'camera': {'lens_mm': LENS, 'sensor_width_mm': SENSOR, 'hfov_deg': round(hfov, 2),
               'location': list(cam.location), 'height_m': CAM_HEIGHT,
               'subject_distance_m': SUBJECT_DISTANCE,
               'aperture_fstop': cam_data.dof.aperture_fstop,
               'rotation_euler_deg': [round(math.degrees(a), 6) for a in cam.rotation_euler],
               'anamorphic': anamorphic_report},
    'panorama_match': {'visible_px_of_16k': round(16384 * hfov / 360),
                       'plate_real_px': 1920,
                       'pct_of_plate': round(100 * (16384 * hfov / 360) / 1920, 1)},
    'render': {'resolution': list(res), 'samples': scn.cycles.samples,
               'shadow_catcher_pass': True},
}
(ENV_DIR / 'scene_info.json').write_text(json.dumps(info, indent=2))

blend = ENV_DIR / ('night_restaurant_test.blend' if args.test else 'night_restaurant.blend')
bpy.ops.wm.save_as_mainfile(filepath=str(blend))

outfile = ENV_DIR / ('background_test.exr' if args.test else 'background_clean.exr')
scn.render.filepath = str(outfile)
bpy.ops.render.render(write_still=True)
info['output'] = str(outfile)
(ENV_DIR / 'scene_info.json').write_text(json.dumps(info, indent=2))
print('HDRI_SCENE', json.dumps(info))
