"""Render the environment per frame, so an animated scene is not frozen behind the actor.

    blender -b <blend> --python scripts/render_background.py -- --shot <dir> [--frames N] [--samples S]

composite.py loads blender/background_shadow.exr ONCE and reuses it for every frame. With a
static rig that is merely wrong in principle; with animated lights it is visibly wrong, since
the room would hold still while the actor's relighting changes. This writes one EXR per frame.
"""
import bpy, sys, json
from pathlib import Path
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--shot', default=None)
ap.add_argument('--frames', type=int, default=0)
ap.add_argument('--samples', type=int, default=0)
ap.add_argument('--scale', type=int, default=0, help='override render width; height follows 16:9')
args = ap.parse_args(argv)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from shot_paths import Shot
shot = Shot(args.shot)
scn = bpy.context.scene
n = args.frames or shot.frame_count
if args.samples:
    scn.cycles.samples = args.samples
if args.scale:
    scn.render.resolution_x, scn.render.resolution_y = args.scale, round(args.scale * 9 / 16)
# Persistent data keeps the Cycles scene resident between renders and serves STALE lighting
# when stepping frames in one process: every probe came back byte-identical with it on.
scn.render.use_persistent_data = False
ims = scn.render.image_settings
if hasattr(ims, 'media_type'):
    ims.media_type = 'IMAGE'
ims.file_format = 'OPEN_EXR'
ims.color_mode = 'RGBA'
ims.color_depth = '32'
ims.exr_codec = 'ZIP'
ims.color_management = 'FOLLOW_SCENE'
env_dir = Path(bpy.data.filepath).parent
out_dir = env_dir / 'background'
out_dir.mkdir(parents=True, exist_ok=True)
for f in range(n):
    scn.frame_set(f)
    scn.render.filepath = str(out_dir / ('%06d.exr' % f))
    bpy.ops.render.render(write_still=True)
    if f % 25 == 0:
        print('BG_FRAME', f, flush=True)
meta = {'shot': shot.name, 'frames': n,
        'resolution': [scn.render.resolution_x, scn.render.resolution_y],
        'samples': scn.cycles.samples, 'dir': str(out_dir)}
(out_dir / 'background.json').write_text(json.dumps(meta, indent=2))
print('BACKGROUND_RENDERED', json.dumps(meta))
