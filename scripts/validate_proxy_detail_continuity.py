"""Measure the depth step the proxy makes where the photographic matte ends.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/validate_proxy_detail_continuity.py \
        --shot <dir> --candidate-dir <alternate proxy dir>

`make_proxy_mesh.build` grows geometry PAST the matte on purpose -- an un-dilated shell loses a
one-cell rim around the whole silhouette. But the fine relief added to that shell comes from
`highpass(integrate_normals(...), matte, ...)`, which is masked BY the matte, so it fell to zero
exactly at the silhouette. A shell that is smooth inside and abruptly flat outside has a ridge
along the matte boundary, one grid cell wide, and its facets catch the light as the dotted rim
the artist saw on the image-left shoulder in 000079 and on the hand in 000051.

This measures that ridge directly, on the geometry, before anything is rendered: the depth
difference between adjacent grid samples whose edge CROSSES the matte boundary.

The number is meaningless on its own -- a real shoulder has real slope there -- so it is reported
against two controls:

* `interior`, the same statistic over edges with both endpoints inside the matte. That is the
  ordinary adjacent-sample depth change on a surface with no artefact, and it is what a fixed
  crossing edge should look like.
* the baseline mesh, so the change is attributable.

It also checks the claim that a detail fix leaves the photographed region untouched, by diffing
every vertex inside the matte between the two builds. "Inside" is decided the way `build` decides
it -- on the matte resized to the DEPTH grid -- because that is the mask whose edge the fix moves.
Judged against the full-resolution matte instead, a handful of boundary vertices do move, and
that gap is reported rather than hidden: geometry calls a pixel inside at 768x432 while the
composite calls it inside at 1920x1080, so a band about 2.5 plate pixels wide around the whole
silhouette is extrapolated geometry multiplied into photographed pixels. That band is where the
dotted rim lives, so the disagreement is a lead, not a rounding detail.
"""
import os
# The mattes are EXR; opencv refuses to read them without this, set before the import.
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args


def front_edges(faces, n_front):
    """Unique undirected edges of the front surface grid, as index pairs."""
    q = faces[np.all(faces < n_front, axis=1)]
    e = np.concatenate([q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 3]], q[:, [3, 0]]], 0)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def sample_alpha(uv, alpha):
    """Front vertices carry their own image position as UV, so this is exact by construction."""
    h, w = alpha.shape
    x = np.clip((uv[:, 0] * w - 0.5).round().astype(int), 0, w - 1)
    y = np.clip(((1.0 - uv[:, 1]) * h - 0.5).round().astype(int), 0, h - 1)
    return alpha[y, x]


def stats(mesh, matte):
    verts, faces = mesh['verts'].astype(np.float64), mesh['faces']
    n_front = int(mesh['n_front'])
    inside = sample_alpha(mesh['uv'][:n_front], matte.astype(np.float32)) > 0.5
    e = front_edges(faces, n_front)
    a, b = e[:, 0], e[:, 1]
    # +Y is depth away from camera. The jump is the step in the surface, in millimetres.
    jump = np.abs(verts[a, 1] - verts[b, 1]) * 1000.0
    crossing = inside[a] != inside[b]
    interior = inside[a] & inside[b]
    out = {}
    for name, sel in (('crossing_matte', crossing), ('interior', interior)):
        j = jump[sel]
        out[name] = {'edges': int(sel.sum()),
                     'median_mm': round(float(np.median(j)), 2) if j.size else None,
                     'p95_mm': round(float(np.percentile(j, 95)), 2) if j.size else None}
    return out, inside, verts, n_front


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--candidate-dir', required=True, help='alternate proxy mesh directory')
    ap.add_argument('--baseline-dir', default='', help='defaults to the shot production proxy dir')
    ap.add_argument('--frames', default='', help='comma-separated; defaults to what both dirs share')
    args = ap.parse_args()
    shot = from_args(args)

    base_dir = Path(args.baseline_dir) if args.baseline_dir else shot.relight / 'proxy'
    cand_dir = Path(args.candidate_dir)
    have = sorted({int(p.stem) for p in cand_dir.glob('*.npz')} &
                  {int(p.stem) for p in base_dir.glob('*.npz')})
    frames = [int(f) for f in args.frames.split(',')] if args.frames else have
    if not frames:
        raise SystemExit('No frames present in both %s and %s' % (base_dir, cand_dir))

    per_frame = []
    for fi in frames:
        alpha = cv2.imread(str(shot.alpha(fi)), cv2.IMREAD_UNCHANGED)
        if alpha is None:
            raise SystemExit('No matte for frame %d' % fi)
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        alpha = np.clip(alpha.astype(np.float32) /
                        {np.dtype(np.uint16): 65535.0, np.dtype(np.uint8): 255.0}
                        .get(alpha.dtype, 1.0), 0, 1)
        # `build` thresholds the matte after resizing it to the depth grid. Use the same mask, or
        # the boundary cell is classified differently here than by the code under test.
        dh, dw = np.load(shot.relight / 'depth' / ('%06d.npy' % fi)).shape[:2]
        matte = np.clip(cv2.resize(alpha, (dw, dh), interpolation=cv2.INTER_AREA), 0, 1) > 0.5
        base, cand = np.load(base_dir / ('%06d.npz' % fi)), np.load(cand_dir / ('%06d.npz' % fi))
        if base['verts'].shape != cand['verts'].shape:
            raise SystemExit('Frame %d: meshes differ in topology, not comparable' % fi)
        b_stat, inside, b_verts, n_front = stats(base, matte)
        c_stat, _, c_verts, _ = stats(cand, matte)
        inside_plate = sample_alpha(base['uv'][:n_front], alpha) > 0.5
        # Does the candidate leave the photographed region alone? Padding is allowed to move;
        # a vertex under the matte is a measured surface and must not.
        d = np.abs(c_verts[:n_front] - b_verts[:n_front])
        per_frame.append({
            'frame': fi,
            'baseline': b_stat, 'candidate': c_stat,
            'max_vertex_move_inside_matte_mm': round(float(d[inside].max() * 1000), 4),
            'max_vertex_move_in_padding_mm': round(float(d[~inside].max() * 1000), 2)
                if (~inside).any() else None,
            # The two mattes disagree over a thin band. Those vertices are outside the geometry
            # matte, so the fix is free to move them, but the composite treats them as subject.
            'matte_resolution_disagreement': {
                'vertices': int((inside_plate & ~inside).sum()),
                'max_move_mm': round(float(d[inside_plate & ~inside].max() * 1000), 2)
                    if (inside_plate & ~inside).any() else None},
            'front_verts': int(n_front), 'inside_matte': int(inside.sum())})

    def med(rows, which, key):
        v = [r[which][key]['median_mm'] for r in rows if r[which][key]['median_mm'] is not None]
        return round(float(np.mean(v)), 2) if v else None

    summary = {
        'shot': shot.name,
        'baseline_dir': str(base_dir), 'candidate_dir': str(cand_dir),
        'measures': 'depth step in mm between adjacent front-grid samples, +Y = away from camera',
        'result': {
            'baseline_crossing_median_mm': med(per_frame, 'baseline', 'crossing_matte'),
            'candidate_crossing_median_mm': med(per_frame, 'candidate', 'crossing_matte'),
            'interior_control_median_mm': med(per_frame, 'baseline', 'interior'),
            'max_vertex_move_inside_matte_mm':
                round(max(r['max_vertex_move_inside_matte_mm'] for r in per_frame), 4),
            'max_move_in_matte_resolution_disagreement_mm':
                max((r['matte_resolution_disagreement']['max_move_mm'] for r in per_frame
                     if r['matte_resolution_disagreement']['max_move_mm'] is not None),
                    default=None)},
        'reading': ('The fix works if the candidate crossing figure falls toward the interior '
                    'control, and if no vertex under the matte moved at all.'),
        'per_frame': per_frame}
    shot.log('proxy_detail_continuity.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('result', 'reading')}, indent=2))


if __name__ == '__main__':
    main()
