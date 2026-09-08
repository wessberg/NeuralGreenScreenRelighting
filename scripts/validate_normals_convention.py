"""Unit-test the Sapiens2 camera-space normal convention against known geometry.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/validate_normals_convention.py

Neural normal models emit camera-space unit vectors, but *which* camera space is inherited from
whatever training rig produced the checkpoint and is written down nowhere in the model card, the
config or the paper. The candidates differ by an axis swap and up to three sign flips: 48 signed
permutations, of which 24 are rotations and 24 are reflections.

A reflection is the dangerous case, and it is why this file exists. Reflecting across a diagonal
leaves that diagonal invariant, so a spot-check along one direction passes while every other
direction is wrong -- the same trap ``validate_sh_convention.py`` was written for on the probe
side. The remedy is the same: pin each axis with its own independent measurement.

Four tests, deliberately not one:

* **Sphere.** An analytic sphere has exact ground-truth normals at every pixel, so all 48
  permutations can be ranked by mean angular error outright. Being geometry rather than a
  person it is out of distribution for a human-specialised model, so it is corroborated, never
  trusted alone.
* **Horizontal flip.** Mirroring the input must negate the horizontal channel and leave the
  other two alone. In distribution (the config ships ``flip_pairs``, so flip augmentation was
  part of training), and it pins the horizontal axis on the real plate.
* **Vertical flip.** Same argument for the vertical axis. An upside-down person is out of
  distribution, so this one is expected to be noisier -- reported with its correlation so the
  weaker evidence is visible rather than hidden.
* **Anatomy.** On the real plate: the left of the face must lean left and the right lean right,
  the forehead must lean up relative to the jaw, and the mean of a visible surface must point
  back at the camera. Three more independent directions, all in distribution.

Camera space becomes Blender world space through the shot's own camera orientation, which is
derived here from the recorded Euler angles rather than asserted, then written with the full
evidence to ``logs/normals_convention.json``. ``neural_normals.py`` refuses to run without it.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, math, argparse, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args
from neural_normals import Sapiens2Normals, matte_bbox, prepare, IN_H, IN_W

AXES = ('x', 'y', 'z')


def signed_permutations():
    """All 48 signed axis permutations, rotations and reflections alike."""
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3), np.float32)
            for row, (src, s) in enumerate(zip(perm, signs)):
                M[row, src] = s
            yield M


def label(M):
    out = []
    for row in M:
        i = int(np.argmax(np.abs(row)))
        out.append(('+' if row[i] > 0 else '-') + AXES[i])
    return ' '.join(out)


def camera_basis(euler_xyz_deg):
    """Blender camera axes in world space, from the recorded Euler rotation.

    Derived, not asserted: a camera looks down its local -Z with local +Y up, so the world
    directions follow from the rotation matrix alone. Blender's default XYZ Euler order.
    """
    rx, ry, rz = (math.radians(a) for a in euler_xyz_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return {'right': R @ np.array([1., 0, 0]),
            'up': R @ np.array([0., 1, 0]),
            'toward_camera': R @ np.array([0., 0, 1])}      # local +Z points back at the viewer


def analytic_sphere(radius_px=300.0):
    """A shaded sphere plus its exact image-space normals (+x right, +y down, +z at camera)."""
    yy, xx = np.indices((IN_H, IN_W), dtype=np.float32)
    u, v = (xx - IN_W / 2) / radius_px, (yy - IN_H / 2) / radius_px
    r2 = u * u + v * v
    inside = r2 < 1.0
    gt = np.stack([u, v, np.sqrt(np.clip(1 - r2, 0, 1))], -1)
    gt[~inside] = 0
    L = np.array([0.4, -0.5, 0.76]); L /= np.linalg.norm(L)
    shade = np.clip((gt * L).sum(-1), 0, 1) * 0.75 + 0.12
    img = np.where(inside[..., None], np.repeat(shade[..., None], 3, -1), 0.5)
    return img.astype(np.float32), gt, inside & (r2 < 0.81)     # trust the interior, not the rim


def load_plate(shot, i):
    fg = cv2.imread(str(shot.fg(i)), -1)[..., ::-1].astype(np.float32)
    alpha = cv2.imread(str(shot.alpha(i)), -1)
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    alpha = np.clip(alpha, 0, 1)
    box = matte_bbox(alpha)
    a_model = cv2.resize(alpha[box[1]:box[1] + box[3], box[0]:box[0] + box[2]],
                         (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    return prepare(fg, alpha, box), a_model, box, alpha.shape


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--frame', type=int, default=0)
    args = ap.parse_args()
    shot = from_args(args)
    model = Sapiens2Normals()
    report = {'model': 'facebook/sapiens2-normal-1b', 'shot': shot.name, 'tests': {}}

    # ---- Test 1: analytic sphere, exact ground truth -----------------------------------
    img, gt, core = analytic_sphere()
    n = model(img)
    ranked = []
    for M in signed_permutations():
        err = np.degrees(np.arccos(np.clip((n[core] @ M.T * gt[core]).sum(-1), -1, 1))).mean()
        ranked.append((float(err), label(M), M))
    ranked.sort(key=lambda r: r[0])
    best_err, best_label, best_M = ranked[0]
    report['tests']['sphere'] = {
        'ground_truth': 'analytic unit sphere, image axes +x right / +y down / +z toward camera',
        'pixels': int(core.sum()),
        'best': {'mapping': best_label, 'mean_angular_error_deg': round(best_err, 3)},
        'runner_up': {'mapping': ranked[1][1], 'mean_angular_error_deg': round(ranked[1][0], 3)},
        'margin_deg': round(ranked[1][0] - best_err, 3),
        'worst_deg': round(ranked[-1][0], 3),
        'all_48': [{'mapping': l, 'err_deg': round(e, 2)} for e, l, _ in ranked]}

    # The sphere's truth is in image axes, so best_M maps model output -> image axes.
    to_image = best_M

    # ---- Tests 2 and 3: flip equivariance on the real plate ----------------------------
    img0, a_model, box, frame_shape = load_plate(shot, args.frame)
    n0 = model(img0)
    m = a_model > 0.99
    for name, flip_back, expect in (('horizontal_flip', lambda a: a[:, ::-1], 0),
                                    ('vertical_flip', lambda a: a[::-1], 1)):
        nf = flip_back(model(np.ascontiguousarray(flip_back(img0))))
        corr = [float(np.corrcoef(n0[m][:, k], nf[m][:, k])[0, 1]) for k in range(3)]
        # In image axes, so the negated channel names the flipped image axis directly.
        img_corr = [float(np.corrcoef((n0 @ to_image.T)[m][:, k],
                                      (nf @ to_image.T)[m][:, k])[0, 1]) for k in range(3)]
        negated = int(np.argmin(img_corr))
        report['tests'][name] = {
            'expectation': 'exactly the %s image axis negates' % ('horizontal', 'vertical')[expect],
            'per_channel_corr_model_axes': [round(c, 3) for c in corr],
            'per_channel_corr_image_axes': [round(c, 3) for c in img_corr],
            'negated_image_axis': AXES[negated],
            'agrees': bool(negated == expect),
            'note': ('in distribution: the config ships flip_pairs, so horizontal flip '
                     'augmentation was part of training'
                     if expect == 0 else
                     'OUT of distribution: an upside-down person is unlike anything in the '
                     'training set, so a weaker correlation here is expected and is not '
                     'evidence against the mapping')}

    # ---- Test 4: anatomy, three more independent directions ----------------------------
    track = json.loads((shot.relight / 'face_track.json').read_text())
    row = next(r for r in track['per_frame'] if r.get('found') and r['frame'] == args.frame)
    fx0, fy0, fx1, fy1 = row['bbox_px']
    sx, sy = IN_W / box[2], IN_H / box[3]
    bx0, by0 = (fx0 - box[0]) * sx, (fy0 - box[1]) * sy
    bx1, by1 = (fx1 - box[0]) * sx, (fy1 - box[1]) * sy
    n_img = n0 @ to_image.T

    def region(x_lo, x_hi, y_lo, y_hi):
        xs = slice(int(bx0 + (bx1 - bx0) * x_lo), max(int(bx0 + (bx1 - bx0) * x_hi), 1))
        ys = slice(int(by0 + (by1 - by0) * y_lo), max(int(by0 + (by1 - by0) * y_hi), 1))
        sub, sm = n_img[ys, xs], m[ys, xs]
        return sub[sm].mean(0) if sm.any() else np.zeros(3)

    left, right = region(.0, .28, .25, .75), region(.72, 1., .25, .75)
    brow, jaw = region(.3, .7, .0, .2), region(.3, .7, .8, 1.)
    checks = [
        ('face left leans left (image -x)', float(left[0]), '<', 0.0),
        ('face right leans right (image +x)', float(right[0]), '>', 0.0),
        ('brow leans up relative to jaw (image -y)', float(brow[1] - jaw[1]), '<', 0.0),
        ('visible surface faces the camera (image +z)', float(n_img[m][:, 2].mean()), '>', 0.3),
    ]
    report['tests']['anatomy'] = {
        'face_bbox_px': row['bbox_px'],
        'checks': [{'test': t, 'value': round(v, 4), 'requires': '%s %.2f' % (op, ref),
                    'pass': bool(v < ref if op == '<' else v > ref)} for t, v, op, ref in checks],
        'all_pass': all((v < ref if op == '<' else v > ref) for _, v, op, ref in checks)}

    # ---- camera space -> Blender world -------------------------------------------------
    # Prefer a rotation the scene builder actually recorded. The fallback matches
    # build_hdri_scene.py's (90, 0, 0), but a fallback that silently agrees is not evidence, so
    # which source was used is reported alongside the result.
    euler, euler_src = None, None
    for name in ('scene_info.json', 'lighting_sh.json'):
        f = shot.environment / name
        if f.exists():
            c = (json.loads(f.read_text()).get('camera') or {})
            if c.get('rotation_euler_deg'):
                euler, euler_src = c['rotation_euler_deg'], name
                break
    if euler is None:
        euler, euler_src = [90.0, 0.0, 0.0], 'DEFAULT (build_hdri_scene.py convention, not recorded)'
    basis = camera_basis(euler)
    # Rows of image_to_world map (image-right, image-down, toward-camera) onto world axes.
    image_to_world = np.stack([basis['right'], -basis['up'], basis['toward_camera']], axis=1)
    M_world = image_to_world @ to_image
    M_world = np.round(M_world, 6)

    report['camera'] = {'rotation_euler_deg': list(euler), 'rotation_source': euler_src,
                        'world_right': basis['right'].round(6).tolist(),
                        'world_up': basis['up'].round(6).tolist(),
                        'world_toward_camera': basis['toward_camera'].round(6).tolist(),
                        'derivation': 'from the recorded Euler angles: a Blender camera looks '
                                      'down local -Z with local +Y up'}
    report['model_to_image'] = {'matrix': to_image.tolist(), 'mapping': best_label}
    report['matrix'] = M_world.tolist()
    report['mapping'] = ('world = (%s)' % ', '.join(
        ' '.join(('%+g*c%d' % (v, i)) for i, v in enumerate(r) if v) or '0' for r in M_world))
    all_pass = (report['tests']['sphere']['margin_deg'] > 5.0
                and report['tests']['horizontal_flip']['agrees']
                and report['tests']['vertical_flip']['agrees']
                and report['tests']['anatomy']['all_pass'])
    report['verdict'] = 'PASS' if all_pass else 'FAIL'

    out = ROOT / 'logs' / 'normals_convention.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print('sphere      best %-11s %.3f deg   runner-up %-11s %.3f deg   margin %.2f'
          % (best_label, best_err, ranked[1][1], ranked[1][0],
             report['tests']['sphere']['margin_deg']))
    for t in ('horizontal_flip', 'vertical_flip'):
        r = report['tests'][t]
        print('%-11s negates image %s  %s   corr %s'
              % (t.split('_')[0], r['negated_image_axis'], 'OK' if r['agrees'] else 'MISMATCH',
                 r['per_channel_corr_image_axes']))
    for c in report['tests']['anatomy']['checks']:
        print('anatomy     %-44s %+.4f  %s' % (c['test'], c['value'], 'OK' if c['pass'] else 'FAIL'))
    print('camera->world matrix %s   %s' % (M_world.tolist(), report['mapping']))
    print('VERDICT %s  ->  %s' % (report['verdict'], out))
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
