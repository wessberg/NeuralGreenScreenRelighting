"""Animate a stress-test lighting rig: level, hue, direction, proximity and beat.

    blender -b <blend> --python scripts/animate_lights.py -- --shot <dir> [--bpm 120]

Purpose. A static rig cannot show whether relighting tracks the scene: a flat gain and a
correct one look identical when nothing changes. This drives every axis a relighter can get
wrong, so failures become visible rather than arguable:

  level        the world dims from bright to near-dark across the shot (~50x)
  hue          an accent sweeps the colour wheel
  direction    a key orbits the subject through a known azimuth
  proximity    a practical travels 2.40 m -> 0.45 m, an inverse-square swing of ~28x
  temporal     the practical pulses on the beat, exposing lag in any recursive filter

IMPORTANT: the emitters are EMISSIVE MESHES, not Blender light objects. Light objects are
invisible to camera rays -- visible_camera defaults to False and forcing it True still renders
nothing -- so a panoramic light probe cannot see them and captures only the world. Every probe
came back byte-identical until this was found. Emissive geometry is captured correctly, is
still sampled as a light by Cycles, and is more honest anyway: a real practical is a visible
object in the room.

Cues are expressed in beats, because the eventual purpose is lighting driven by a song.
Everything is keyframed on real scene data, so it reaches the actor through the baked probes
automatically -- there is no separate "actor lighting" to keep in sync.
"""
import bpy, bmesh, sys, json, math, colorsys
from pathlib import Path

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--shot', default=None)
ap.add_argument('--bpm', type=float, default=120.0)
ap.add_argument('--sweep-degrees', type=float, default=200.0)
ap.add_argument('--out', default=None)
args = ap.parse_args(argv)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from shot_paths import Shot

shot = Shot(args.shot)
cfg = shot.config
cam_cfg = cfg.get('camera') or {}
CAM_H = float(cam_cfg.get('height_m', 1.55))
n_frames = shot.frame_count
fps = float(shot.frame_rate)
bps = args.bpm / 60.0

scn = bpy.context.scene
scn.frame_start, scn.frame_end = 0, n_frames - 1
# render.fps is an int16, so a numerator like 60000 is silently clamped to 32767. Blender's
# convention for fractional rates is a small integer with a fractional base: 60 / 1.001.
scn.render.fps = int(round(fps))
scn.render.fps_base = scn.render.fps / fps

for ob in [o for o in bpy.data.objects if o.get('vp_animated')]:
    bpy.data.objects.remove(ob, do_unlink=True)

SUBJECT = (0.0, 0.0, CAM_H - 0.10)


def emissive_panel(name, size, color, strength):
    """A camera-visible emissive plane, which a light probe can actually see."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size / 2.0)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob['vp_animated'] = True
    scn.collection.objects.link(ob)
    mat = bpy.data.materials.new(name + ' emission')
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type != 'OUTPUT_MATERIAL':
            nt.nodes.remove(n)
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Color'].default_value = (color[0], color[1], color[2], 1.0)
    em.inputs['Strength'].default_value = strength
    nt.links.new(em.outputs['Emission'], nt.nodes['Material Output'].inputs['Surface'])
    if hasattr(mat, 'cycles'):
        mat.cycles.emission_sampling = 'FRONT'
    ob.data.materials.append(mat)
    return ob, em


key, key_em = emissive_panel('Sweep key', 1.5, (0.62, 0.74, 1.00), 12.0)
practical, prac_em = emissive_panel('Beat practical', 0.35, (1.00, 0.52, 0.22), 60.0)
accent, accent_em = emissive_panel('Hue accent', 0.9, (1.00, 0.35, 0.75), 8.0)

RADIUS, KEY_H = 2.4, CAM_H + 0.35

world = scn.world
bg_node = next((n for n in world.node_tree.nodes if n.type == 'BACKGROUND'), None)
if bg_node is None:
    raise SystemExit('World has no Background node; run build_hdri_scene.py first')


def aim(ob, target):
    d = (target[0] - ob.location[0], target[1] - ob.location[1], target[2] - ob.location[2])
    ob.rotation_euler = (math.atan2(math.hypot(d[0], d[1]), -d[2]), 0.0, math.atan2(-d[0], d[1]))


cues = []
for f in range(n_frames):
    t = f / fps
    beat = t * bps
    frac = f / max(n_frames - 1, 1)

    # LEVEL: the world dims from bright to near dark, so the actor must track a large
    # environment level change rather than a fixed exposure.
    env_strength = 1.6 * (0.02 + 0.98 * (0.5 * (1 + math.cos(math.pi * frac)) ** 1.6))
    bg_node.inputs['Strength'].default_value = env_strength
    bg_node.inputs['Strength'].keyframe_insert('default_value', frame=f)

    # DIRECTION: key orbits through a known azimuth, passing through frontal, so both sides
    # of the face are keyed at some point in the shot.
    az = math.radians(-args.sweep_degrees / 2 + args.sweep_degrees * frac)
    key.location = (SUBJECT[0] + RADIUS * math.sin(az), SUBJECT[1] - RADIUS * math.cos(az), KEY_H)
    aim(key, SUBJECT)
    key.keyframe_insert('location', frame=f)
    key.keyframe_insert('rotation_euler', frame=f)
    key_em.inputs['Strength'].default_value = 6.0 + 22.0 * (1 - abs(2 * frac - 1))
    key_em.inputs['Strength'].keyframe_insert('default_value', frame=f)

    # PROXIMITY: the practical travels in and back out, a ~28x inverse-square swing.
    d_prox = 2.4 - 1.95 * (0.5 * (1 - math.cos(2 * math.pi * frac)))
    practical.location = (0.55 * d_prox, -0.35 * d_prox, CAM_H + 0.05)
    aim(practical, SUBJECT)
    practical.keyframe_insert('location', frame=f)
    practical.keyframe_insert('rotation_euler', frame=f)
    # TEMPORAL: sharp attack, exponential decay, once per beat.
    pulse = math.exp(-4.0 * (beat % 1.0))
    prac_em.inputs['Strength'].default_value = 18.0 + 55.0 * pulse
    prac_em.inputs['Strength'].keyframe_insert('default_value', frame=f)

    # HUE: the accent sweeps the colour wheel, so the colour of light on the actor must follow.
    r, g, b = colorsys.hsv_to_rgb((0.88 + 0.55 * frac) % 1.0, 0.85, 1.0)
    accent_em.inputs['Color'].default_value = (r, g, b, 1.0)
    accent_em.inputs['Color'].keyframe_insert('default_value', frame=f)
    accent.location = (-1.55, 0.45 + 0.5 * math.sin(2 * math.pi * 0.4 * t), CAM_H - 0.2)
    aim(accent, SUBJECT)
    accent.keyframe_insert('location', frame=f)
    accent.keyframe_insert('rotation_euler', frame=f)
    accent_em.inputs['Strength'].default_value = 4.0 + 16.0 * frac
    accent_em.inputs['Strength'].keyframe_insert('default_value', frame=f)

    if f % 25 == 0:
        cues.append({'frame': f, 'time_s': round(t, 4), 'beat': round(beat, 3),
                     'env_strength': round(env_strength, 4),
                     'key_azimuth_deg': round(math.degrees(az), 1),
                     'key_strength': round(key_em.inputs['Strength'].default_value, 2),
                     'practical_distance_m': round(d_prox, 3),
                     'practical_strength': round(prac_em.inputs['Strength'].default_value, 2),
                     'accent_rgb': [round(r, 3), round(g, 3), round(b, 3)]})


def iter_fcurves(holder):
    """Blender 5.x replaced action.fcurves with slotted actions: layers -> strips ->
    channelbag(slot) -> fcurves. Handle both so this runs on 4.x and 5.x."""
    ad = getattr(holder, 'animation_data', None)
    if not ad or not ad.action:
        return
    act = ad.action
    if hasattr(act, 'fcurves'):
        yield from act.fcurves
        return
    slot = getattr(ad, 'action_slot', None)
    for layer in getattr(act, 'layers', []):
        for strip in layer.strips:
            bag = None
            if slot is not None:
                try:
                    bag = strip.channelbag(slot)
                except Exception:
                    bag = None
            if bag is not None:
                yield from bag.fcurves
            else:
                for bag2 in getattr(strip, 'channelbags', []):
                    yield from bag2.fcurves


for holder in (key, practical, accent,
               key.data.materials[0].node_tree,
               practical.data.materials[0].node_tree,
               accent.data.materials[0].node_tree,
               world.node_tree):
    for fc in iter_fcurves(holder):
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'

meta = {
    'shot': shot.name, 'frames': n_frames, 'fps': str(shot.frame_rate),
    'duration_s': round(n_frames / fps, 4), 'bpm': args.bpm,
    'sweep_degrees': args.sweep_degrees,
    'emitters': ('EMISSIVE MESHES, not light objects: Blender lights are invisible to camera '
                 'rays, so a light probe cannot see them'),
    'axes_stressed': {
        'level': 'world strength 1.63 -> 0.03, roughly a 50x environment level change',
        'direction': 'key orbits %.0f degrees of azimuth' % args.sweep_degrees,
        'proximity': 'practical travels 2.40 m -> 0.45 m, about a 28x inverse-square swing',
        'hue': 'accent sweeps the colour wheel',
        'temporal': 'practical pulses exp(-4*phase) once per beat at %.0f BPM' % args.bpm,
    },
    'sampled_cues': cues,
}
env_dir = Path(bpy.data.filepath).parent
(env_dir / 'light_animation.json').write_text(json.dumps(meta, indent=2))
out = Path(args.out) if args.out else Path(bpy.data.filepath)
bpy.ops.wm.save_as_mainfile(filepath=str(out))
print('ANIMATED_LIGHTS', json.dumps({'blend': str(out), 'frames': n_frames,
                                     'axes': list(meta['axes_stressed'])}))
