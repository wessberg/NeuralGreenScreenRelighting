"""Build proxy geometry for the keyed performer, so Cycles can ray-trace against them.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/make_proxy_mesh.py --shot <dir>

A gain map multiplies the plate, so it can only redistribute light the plate already shows.
Shading is dot(n, l); shadowing is "is the path to that light blocked?". No normal map answers
the second question -- a nose casts a shadow onto a cheek only if a ray from the cheek toward
the light hits the nose. That needs geometry, which is what this stage makes.

The proxy is a **displaced silhouette shell**:

* the matte gives the silhouette, which is the most visually important property of a shadow,
  and the key already produces an accurate one;
* metric depth displaces it, using the same calibration and the same back-projection as
  ``relight_sh.normals_from_depth`` so proxy and normals cannot disagree about where the actor
  is;
* a **back shell** closes it. Without one the mesh is a single open surface, which is invisible
  to shadow rays arriving from behind and would let a back light shine straight through the
  performer's head.

What this proxy is NOT: a real head. It has no eye sockets at the right depth, no ears in the
round, no hair strands, and its back is invented. Expect self-shadowing to be right in the large
-- nose, brow, jaw, chin, the shapes that read -- and wrong in the fine detail. That is a far
higher ceiling than a gain map, not an unlimited one, and it should be judged on a rendered
uplight rather than on a metric.

Output is an npz of vertices and faces in Blender world coordinates, which ``render_proxy.py``
turns into a mesh with ``from_pydata``. Writing geometry rather than an OBJ or PLY keeps the
exchange in one format the repo already reads and avoids an importer's axis conventions, which
is exactly the class of silent error this project keeps finding.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from scipy.ndimage import distance_transform_edt
from shot_paths import add_shot_argument, from_args

W = H = None        # set per shot from the plate aspect; see Shot.working_resolution


def depth_calibration(shot, cfg, dist):
    """The same two-constraint fit relight_sh.py uses: face anchor + torso depth spread.

    Read from the relight log when it exists so the proxy and the gain map cannot drift apart;
    refitted identically otherwise.
    """
    log = shot.log('relight_sh.json')
    if log.exists():
        cal = json.loads(log.read_text()).get('depth_calibration')
        if cal and 'a' in cal:
            return cal, 'reused from logs/relight_sh.json'
    track = json.loads((shot.relight / 'face_track.json').read_text())
    rows = [r for r in track['per_frame'] if r.get('found')]
    mid = rows[len(rows) // 2]
    d0 = np.load(shot.depth_npy(len(rows) // 2))
    a0 = np.clip(cv2.resize(cv2.imread(str(shot.alpha(len(rows) // 2)), -1), (W, H)), 0, 1)
    src_w, src_h = cfg.get('source_resolution', [1920, 1080])
    fu, fv = mid['face_centroid_px']
    fx_px, fy_px = int(fu * W / src_w), int(fv * H / src_h)
    d_face = float(np.median(d0[max(fy_px - 6, 0):fy_px + 7, max(fx_px - 6, 0):fx_px + 7]))
    body = d0[a0 > .99]
    d_lo, d_hi = np.percentile(body, [10, 90]) if body.size else (d_face - 1, d_face + 1)
    spread = 0.26
    a_fit = (1.0 / (dist - spread / 2) - 1.0 / (dist + spread / 2)) / max(d_hi - d_lo, 1e-6)
    return {'a': float(a_fit), 'b': float(1.0 / dist - a_fit * d_face), 'd_face': d_face,
            'torso_spread_m': spread}, 'fitted here'


FACE_SURFACE_DEPTH_M = 0.085
"""Visible-surface depth of an adult face at near-frontal yaw: nose tip to the cheek plane.

Anthropometric, and used as the proxy's depth SCALE anchor. A monocular depth model recovers
depth ORDER reliably and metric relief badly, so the raw fit produced a 0.45 m bust -- about
1.8x too deep, which would have thrown correspondingly exaggerated self-shadows. Anchoring the
scale on the one region whose true depth is known, and whose pixels are the ones that matter for
a face shadow, is a stated assumption rather than a tuned constant."""


def relief_scale(Yw, mask, face_box, target=FACE_SURFACE_DEPTH_M):
    """Factor that makes the face region's p5-p95 depth spread equal `target`.

    Measured on the face rather than the whole bust because the bust contains a microphone held
    well forward of the body, which is real geometry but not a usable scale reference.
    """
    x0, y0, x1, y1 = face_box
    sel = np.zeros(mask.shape, bool)
    sel[y0:y1, x0:x1] = True
    sel &= mask
    if sel.sum() < 200:
        return 1.0, 0.0
    lo, hi = np.percentile(Yw[sel], [5, 95])
    measured = float(hi - lo)
    return (target / measured if measured > 1e-4 else 1.0), measured


def integrate_normals(nrm, mask, X, Zw, detail_cutoff_m):
    """Recover fine surface relief from the neural normals, as a high-pass displacement.

    A monocular depth model gets depth ORDER right and fine relief badly: the raw shell renders
    as a mannequin with no nose ridge, so an uplight throws no nose shadow. The normals carry
    exactly the detail the depth lacks, because for a surface Y = f(X, Z) the normal is
    proportional to (fX, -1, fZ) -- so the normal field IS the gradient field of the surface.

    Integrating that gradient recovers relief with correct high frequencies and drifting low
    frequencies (the classic shape-from-shading failure). Depth has the opposite error profile.
    So each is used where it is trustworthy: depth below `detail_cutoff_m`, integrated normals
    above it. Solved as a Poisson equation with Neumann boundaries via DCT, which is exact and
    non-iterative.
    """
    from scipy.fft import dctn, idctn
    ny = np.minimum(nrm[..., 1], -1e-3)              # facing the camera: ny < 0
    fX = np.clip(-nrm[..., 0] / ny, -4.0, 4.0)
    fZ = np.clip(-nrm[..., 2] / ny, -4.0, 4.0)
    # Gradients are per-metre; convert to per-pixel using the local world spacing.
    dX = np.gradient(X, axis=1)
    dZ = np.gradient(Zw, axis=0)
    gx = fX * dX                                     # dY per column step
    gz = fZ * dZ                                     # dY per row step
    m = mask.astype(np.float32)
    gx, gz = gx * m, gz * m
    div = np.gradient(gx, axis=1) + np.gradient(gz, axis=0)
    H_, W_ = div.shape
    dct = dctn(div, type=2, norm='ortho')
    ii, jj = np.indices((H_, W_), dtype=np.float64)
    denom = 2 * (np.cos(np.pi * ii / H_) + np.cos(np.pi * jj / W_) - 2)
    denom[0, 0] = 1.0
    Y = idctn(dct / denom, type=2, norm='ortho')
    Y[~mask] = 0.0
    return Y


def highpass(field, mask, sigma_px):
    """Field minus its foreground-only low-pass, so the matte edge adds no false relief."""
    m = mask.astype(np.float32)
    lo = cv2.GaussianBlur(field * m, (0, 0), sigma_px) / np.maximum(
        cv2.GaussianBlur(m, (0, 0), sigma_px), 1e-6)
    return (field - lo) * m


def build(alpha, depth, cal, fx_small, cam_h, dist, step, thickness, smooth,
          face_box=None, scale=None, normals=None, detail=1.0, detail_cutoff_m=0.05,
          antialias=True):
    """Front shell from depth, flat-capped back shell, stitched along the silhouette.

    Returns vertices (N,3) in Blender world axes and quad faces, plus diagnostics.
    """
    # DILATE the mesh mask beyond the matte. A quad only forms where all four grid corners are
    # inside, so an un-dilated shell loses a one-cell rim around the whole silhouette -- and the
    # composite then has to inpaint a ratio it never rendered. Measured on frame 1 that rim was
    # 6.8% of the matte, 77% of it along the bottom edge where the matte runs off frame, and it
    # showed as dark patches and a vertical seam. Growing the mesh past the matte costs nothing:
    # the extra geometry falls outside the alpha and is never composited.
    matte = alpha > 0.5
    if not matte.any():
        return None
    # Two masks, deliberately. `matte` decides where the DEPTH is trustworthy; `mask` decides
    # where geometry exists. Dilating a single mask pulled background depth into the shell and
    # re-inflated the relief from 0.21 m to 0.42 m -- about twice a real bust, which throws
    # exaggerated shadows between the fingers of a hand held toward camera.
    mask = cv2.dilate(matte.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1) > 0
    # Same normalised convolution as the normals, so proxy and normals share a surface. Weighted
    # by the MATTE, never the dilated mask, so the extra rim is a smooth extrapolation of real
    # subject depth rather than a sample of the background behind her.
    m = matte.astype(np.float32)
    d = cv2.GaussianBlur(depth * m, (0, 0), smooth) / np.maximum(cv2.GaussianBlur(m, (0, 0), smooth), 1e-6)
    Zc = 1.0 / np.maximum(cal['a'] * d + cal['b'], 1e-4)
    Zc = np.clip(Zc, dist - 1.0, dist + 1.0)

    yy, xx = np.indices((H, W), dtype=np.float32)
    X = (xx + .5 - W / 2) * Zc / fx_small
    Zw = cam_h - (yy + .5 - H / 2) * Zc / fx_small
    Yw = Zc - dist                                   # +Y away from camera, 0 at the subject

    # Rescale the relief about the face plane. Depth ORDER survives; the metric scale does not,
    # and an over-deep proxy casts over-long shadows.
    measured = 0.0
    if face_box is not None:
        s_rel, measured = relief_scale(Yw, matte, face_box)
        if scale is not None:
            s_rel = scale
        anchor = float(np.median(Yw[matte]))
        Yw = anchor + (Yw - anchor) * s_rel
    else:
        s_rel = 1.0

    # Fine relief from the normals, high-passed and added to the depth shell.
    detail_amp = 0.0
    if normals is not None and detail > 0:
        Yn = integrate_normals(normals, matte, X, Zw, detail_cutoff_m)
        # Cutoff in pixels: how many pixels span `detail_cutoff_m` at this distance.
        px_per_m = fx_small / max(float(np.median(Zc[matte])), 1e-3)
        sigma_px = max(detail_cutoff_m * px_per_m, 1.0)
        Yd = highpass(Yn, matte, sigma_px)
        # Geometry extends beyond the photographic matte. Zeroing its normal-derived detail
        # there makes a depth step at the matte boundary, whose grid faces light as a dotted
        # ridge. Continue the nearest measured detail into the padding; preserve every real
        # matte sample exactly, including the fine facial relief used for self-shadows.
        nearest = distance_transform_edt(~matte, return_distances=False, return_indices=True)
        Yd = Yd[nearest[0], nearest[1]]
        # Normalise so `detail` is a plain multiplier on a physically-scaled relief rather than
        # an arbitrary gain: the integrated field's units are already metres.
        Yw = Yw + detail * Yd
        detail_amp = float(np.percentile(np.abs(Yd[mask]), 95))

    # Anti-alias BEFORE decimating. `sub` below samples every `step` depth pixels, so the grid can
    # only carry wavelengths longer than 2*step px -- but the relief above is high-passed at about
    # 40 px and therefore holds energy right down to one pixel. Striding that without a low-pass
    # folds it onto the grid pitch itself, and a corrugation at exactly the facet spacing is what
    # catches the light as the dotted rim the artist saw. Measured: it is not render noise (4x the
    # samples changes nothing) and not the depth cliffs, and it sits across the whole subject
    # rather than only at the silhouette, which is what an aliasing mechanism predicts and a
    # boundary mechanism does not.
    #
    # Nothing real is lost: what this removes is precisely what the vertex grid cannot represent.
    if antialias and step > 1:
        Yw = cv2.GaussianBlur(Yw, (0, 0), step * 0.5)

    # REPROJECT. X and Zw above were back-projected from the ORIGINAL depth, and everything
    # since has changed that depth: the relief rescale (x0.455) and the normal-integrated detail.
    # A vertex whose depth moves but whose X and Z do not no longer projects to the pixel it came
    # from. Measured, the shell landed 38 px too high, which left the bottom of the matte
    # uncovered -- the dark patches along the bottom edge -- and the composite inpainted that gap
    # differently every frame, which is the flicker along her arm.
    #
    # Re-derive X and Z from the FINAL depth so every vertex still lands on its source pixel.
    Zc_final = np.maximum(Yw + dist, 1e-3)
    X = (xx + .5 - W / 2) * Zc_final / fx_small
    Zw = cam_h - (yy + .5 - H / 2) * Zc_final / fx_small

    sub = (slice(None, None, step), slice(None, None, step))
    Xs, Ys, Zs, Ms = X[sub], Yw[sub], Zw[sub], mask[sub]
    h, w = Ms.shape

    idx = -np.ones((h, w), np.int64)
    idx[Ms] = np.arange(Ms.sum())
    n_front = int(Ms.sum())
    front = np.stack([Xs[Ms], Ys[Ms], Zs[Ms]], -1)
    # UVs are the vertex's own image position. The shell IS a back-projected pixel grid, so this
    # projection is exact by construction -- no unwrap, no seam, and the plate lands back on the
    # geometry it came from at full plate resolution rather than at vertex density.
    rows, cols = np.where(Ms)
    uv_front = np.stack([(cols * step + 0.5) / W, 1.0 - (rows * step + 0.5) / H], -1)
    # Flat back cap: a constant offset behind the deepest front point. It is never seen; it
    # exists so the shell is closed and opaque to rays arriving from behind.
    back = front.copy()
    back[:, 1] = float(front[:, 1].max()) + thickness
    verts = np.concatenate([front, back], 0)

    quads = []
    ok = Ms[:-1, :-1] & Ms[:-1, 1:] & Ms[1:, :-1] & Ms[1:, 1:]
    r, c = np.where(ok)
    a, b = idx[r, c], idx[r, c + 1]
    cc, dd = idx[r + 1, c + 1], idx[r + 1, c]
    # WINDING. Columns increase world X and rows DECREASE world Z (Zw = cam_h - (row-...)*...),
    # so traversing a->b->cc->dd goes +X then -Z, whose cross product is +Y -- AWAY from the
    # camera. The front cap was therefore back-facing and the back cap inside-out, which leaves
    # a RANDOM_WALK_SKIN volume inverted. Reversed so the front faces -Y, toward the camera.
    quads.append(np.stack([dd, cc, b, a], -1))                       # front, facing -Y
    quads.append(np.stack([a, b, cc, dd], -1) + n_front)             # back, opposite winding

    # Stitch the silhouette. A wall must join TWO ADJACENT boundary vertices front-to-back, so it
    # needs four distinct indices. The previous version emitted [i, i+n, i+n, i] -- two distinct
    # vertices and therefore zero area, which blocks nothing however thick the shell is. The
    # review measured 96 such degenerate walls and 184 unpaired boundary edges on a test case.
    #
    # A grid edge is on the silhouette when exactly ONE of the two cells sharing it produced a
    # quad. Both its endpoints are inside, so the wall is a real quad.
    okp = np.zeros((h + 1, w + 1), bool)
    okp[1:h, 1:w] = ok
    walls = []
    # Horizontal edges (r,c)-(r,c+1): shared by the cells above and below.
    both = Ms[:, :-1] & Ms[:, 1:]
    bnd = both & (okp[:h, 1:w] ^ okp[1:h + 1, 1:w])
    rr, ccc = np.where(bnd)
    if len(rr):
        i0, i1 = idx[rr, ccc], idx[rr, ccc + 1]
        walls.append(np.stack([i0, i1, i1 + n_front, i0 + n_front], -1))
    # Vertical edges (r,c)-(r+1,c): shared by the cells left and right.
    both = Ms[:-1, :] & Ms[1:, :]
    bnd = both & (okp[1:h, :w] ^ okp[1:h, 1:w + 1])
    rr, ccc = np.where(bnd)
    if len(rr):
        i0, i1 = idx[rr, ccc], idx[rr + 1, ccc]
        walls.append(np.stack([i1, i0, i0 + n_front, i1 + n_front], -1))
    faces = np.concatenate(quads + ([np.concatenate(walls, 0)] if walls else []), 0)
    n_walls = int(sum(len(x) for x in walls))
    in_matte = matte[sub][Ms]
    ref = front[in_matte, 1] if in_matte.any() else front[:, 1]
    p1, p99 = np.percentile(ref, [1, 99])
    return {'verts': verts.astype(np.float32), 'faces': faces.astype(np.int64),
            'uv': np.concatenate([uv_front, uv_front], 0).astype(np.float32),
            'n_front': n_front, 'grid': (h, w),
            'extent_m': (front.max(0) - front.min(0)).tolist(),
            'depth_relief_m': float(front[:, 1].max() - front[:, 1].min()),
            'depth_relief_p1_99_m': float(p99 - p1),
            'relief_scale': float(s_rel), 'face_depth_measured_m': float(measured),
            'detail_p95_m': detail_amp, 'n_walls': n_walls}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--only-frames', default='', help='comma-separated local frames for a diagnostic build')
    ap.add_argument('--output-dir', default='', help='alternate mesh directory; keeps production meshes and log intact')
    ap.add_argument('--step', type=int, default=2, help='pixel stride; 2 gives a 384x216 grid')
    ap.add_argument('--thickness', type=float, default=0.30,
                    help='metres from the deepest front point to the flat back cap')
    ap.add_argument('--smooth', type=float, default=3.0)
    ap.add_argument('--scale', type=float, default=None,
                    help='override the face-anchored depth relief scale')
    ap.add_argument('--detail', type=float, default=1.0,
                    help='weight on normal-integrated fine relief; 0 disables it')
    ap.add_argument('--no-detail-antialias', action='store_true',
                    help='skip the pre-decimation low-pass, to reproduce the aliased build')
    ap.add_argument('--detail-cutoff', type=float, default=0.05,
                    help='metres; below this scale relief comes from the normals, above from depth')
    args = ap.parse_args()
    shot = from_args(args)
    global W, H
    W, H = shot.working_resolution

    cfg = shot.config
    cam = cfg.get('camera') or {}
    lens, sensor = float(cam.get('lens_mm', 50)), float(cam.get('sensor_width_mm', 36))
    dist = float(cam.get('actor_distance_m', 1.30))
    cam_h = float(cam.get('height_m', 1.55))
    src_w = cfg.get('source_resolution', [1920, 1080])[0]
    fx_small = (src_w * lens / sensor) * (W / src_w)
    cal, cal_src = depth_calibration(shot, cfg, dist)

    # Face boxes drive the relief scale anchor.
    track = json.loads((shot.relight / 'face_track.json').read_text())
    src_h = cfg.get('source_resolution', [1920, 1080])[1]
    sx, sy = W / src_w, H / src_h
    boxes = {r['frame']: [int(r['bbox_px'][0] * sx), int(r['bbox_px'][1] * sy),
                          int(r['bbox_px'][2] * sx), int(r['bbox_px'][3] * sy)]
             for r in track['per_frame'] if r.get('found')}

    out = Path(args.output_dir) if args.output_dir else shot.relight / 'proxy'
    out.mkdir(parents=True, exist_ok=True)
    n = args.frames or shot.frame_count
    stats = []
    # One scale for the whole shot, fitted on the anchor frame. Per-frame rescaling would make
    # the actor's depth breathe as the depth model's relative relief wandered.
    fixed_scale = args.scale
    frames = [int(v) for v in args.only_frames.split(',')] if args.only_frames else range(args.start, args.start + n)
    for i in frames:
        alpha = np.clip(cv2.resize(cv2.imread(str(shot.alpha(i)), -1), (W, H),
                                   interpolation=cv2.INTER_AREA), 0, 1)
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        depth = np.load(shot.depth_npy(i))
        nrm = None
        if args.detail > 0:
            np_path = shot.relight / 'normals' / ('%06d.npz' % i)
            if np_path.exists():
                nrm = np.load(np_path)['normal'].astype(np.float32)
                if nrm.shape[:2] != (H, W):
                    nrm = cv2.resize(nrm, (W, H), interpolation=cv2.INTER_LINEAR)
                nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
        mesh = build(alpha, depth, cal, fx_small, cam_h, dist, args.step, args.thickness,
                     args.smooth, face_box=boxes.get(i), scale=fixed_scale,
                     normals=nrm, detail=args.detail, detail_cutoff_m=args.detail_cutoff,
                     antialias=not args.no_detail_antialias)
        if mesh is None:
            continue
        if fixed_scale is None and mesh['relief_scale'] != 1.0:
            fixed_scale = mesh['relief_scale']
        np.savez_compressed(out / ('%06d.npz' % i), verts=mesh['verts'], faces=mesh['faces'],
                            n_front=mesh['n_front'], uv=mesh['uv'])
        stats.append({'frame': i, 'verts': int(len(mesh['verts'])),
                      'faces': int(len(mesh['faces'])), 'n_front': mesh['n_front'],
                      'extent_m': [round(v, 4) for v in mesh['extent_m']],
                      'depth_relief_m': round(mesh['depth_relief_m'], 4),
                      'depth_relief_p1_99_m': round(mesh['depth_relief_p1_99_m'], 4),
                      'relief_scale': round(mesh['relief_scale'], 4),
                      'face_depth_before_scale_m': round(mesh['face_depth_measured_m'], 4),
                      'normal_detail_p95_m': round(mesh['detail_p95_m'], 5),
                      'side_wall_quads': mesh['n_walls']})
        if i % 25 == 0:
            print(json.dumps(stats[-1]), flush=True)

    relief = np.array([s['depth_relief_m'] for s in stats])
    ext = np.array([s['extent_m'] for s in stats])
    summary = {'shot': shot.name, 'frames': len(stats),
               'depth_calibration': cal, 'calibration_source': cal_src,
               'stride': args.step, 'back_shell_thickness_m': args.thickness,
               'geometry': 'displaced silhouette shell, flat back cap, stitched side walls',
               'world_axes': '+X right, +Y away from camera, +Z up; actor centred at Y=0',
               'limits': 'no eye sockets at true depth, no ears in the round, no hair strands, '
                         'invented back. Self-shadowing is right in the large and wrong in '
                         'fine detail.',
               'face_surface_depth_target_m': FACE_SURFACE_DEPTH_M,
               'relief_scale_applied': fixed_scale,
               'result': {'mean_depth_relief_p1_99_m':
                              float(np.mean([s['depth_relief_p1_99_m'] for s in stats])),
                          'mean_depth_relief_m': float(relief.mean()),
                          'mean_extent_m': ext.mean(0).round(4).tolist(),
                          'verts_per_frame': int(np.mean([s['verts'] for s in stats]))},
               'per_frame': stats}
    log_path = out / 'proxy_mesh.json' if args.output_dir else shot.log('proxy_mesh.json')
    log_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('frames', 'result')}, indent=2))


if __name__ == '__main__':
    main()
