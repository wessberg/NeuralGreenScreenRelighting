"""DIAGNOSTIC (round 40, for round 41): is this eye's limbus a foreshortened ellipse?

At an extreme sideways glance her iris is foreshortened -- at 961-964 and 1050-1054 left its
dark disc is ~14-16 px wide against the 24.5 px shot diameter -- and the fixed-radius CIRCLE
of limbus_fit.py cannot fit it: it sits ~5 px to one side with half of it on sclera, the lids
fitted around it collapse the opening and the pass drops an open eye with the softbox up
(round40.md section 3). The pass already paints an axis-aligned ellipse when the track gives
ax != ay; what is missing is the fit.

For each listed eye this searches an axis-aligned ellipse -- vertical semi-axis the shot
radius r, horizontal b in 0.45r..r -- over +/-8 px x, +/-3 px y around the tracked centre,
maximising Daugman's mean derivative along the contour NORMAL over the angles clear of the
eyelid polygon (limbus_fit's LID_MIN / MIN_ANGLES), and compares it with the best circle
(b = r) in the same window. Writes nothing a stage reads.

Measured (logs/r40_ellipse_probe_*.json): glances +23..+37 % over the best circle at
b/r ~0.63 (125 left +197 % at the search floor 0.45); ordinary eyes b/r 1.00, gain 0 (p90 of
700 eyes 0.0 %). AMBIGUOUS: the frames beside a glance (954-960, 965, 1049 left) prefer a
narrow ellipse displaced the OTHER way by +3..+8 % -- likely the pupil edge on one side and
the limbus on the other. Not a production stage.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/probe_r40_ellipse.py         "961:left,962:left,700:left" out.json
"""
import json, sys, os
from pathlib import Path
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
import torch
sys.path.insert(0, 'scripts')
import eye_gpu as G
import track_gpu as T
import limbus_fit as lf
from eye_gpu import DEV, F32, as_t

F = Path('music_videos/reporter_test/shots/reporter_full')
lt = json.loads((F / 'relight' / 'limbus_track_tv.json').read_text(encoding='utf-8'))
meta = json.loads((F / 'relight' / 'face_landmarks.json').read_text())
with np.load(F / 'relight' / 'face_landmarks.npz') as z:
    points = z['points']
top = meta['topology']
R = lt['shot_radius']
A = 128
th = torch.arange(A, device=DEV, dtype=torch.float32) * (2 * np.pi / A)
ct, st = torch.cos(th), torch.sin(th)


def search(i, side, span_x=8.0, span_y=3.0, step=0.25, bs=None):
    row = lt['rows'][i]
    e = [x for x in row['eyes'] if x['side'] == side][0]
    c0 = np.array(e.get('smoothed_centre') or e['landmark_centre'], float)
    r = R[side]
    img = cv2.imread(str(F / 'key' / 'fg' / ('%06d.exr' % i)), cv2.IMREAD_UNCHANGED)[..., ::-1]
    reach = r * 1.7 + 10 + span_x
    x0 = int(np.floor(c0[0] - reach)); y0 = int(np.floor(c0[1] - reach))
    x1 = int(np.ceil(c0[0] + reach)) + 1; y1 = int(np.ceil(c0[1] + reach)) + 1
    patch = torch.as_tensor(np.ascontiguousarray(img[y0:y1, x0:x1]), device=DEV).half()[None]
    poly = points[i][top['eyelid_' + side], :2]
    gx, gy, lid = T.limbus_fields(patch, [dict(poly=poly, x0=x0, y0=y0)])
    hw = as_t(np.array([[y1 - y0, x1 - x0]]))
    samp = G.CubicSampler([gx, gy, lid], hw)
    offx = torch.arange(-span_x, span_x + 1e-6, step, device=DEV)
    offy = torch.arange(-span_y, span_y + 1e-6, step, device=DEV)
    if bs is None:
        bs = torch.linspace(0.45, 1.0, 23, device=DEV) * r
    O1, O2, NB = len(offy), len(offx), len(bs)
    cx = (c0[0] - x0) + offx.view(1, O2, 1, 1)
    cy = (c0[1] - y0) + offy.view(O1, 1, 1, 1)
    bb = bs.view(1, 1, NB, 1)
    px = cx + bb * ct.view(1, 1, 1, A)
    py = cy + r * st.view(1, 1, 1, A)
    px, py = torch.broadcast_tensors(px, py)
    sgx, sgy, slid = samp(px.reshape(1, -1).contiguous(), py.reshape(1, -1).contiguous())
    sgx = sgx.view(O1, O2, NB, A); sgy = sgy.view(O1, O2, NB, A); slid = slid.view(O1, O2, NB, A)
    nx = ct.view(1, 1, 1, A) / bb; ny = st.view(1, 1, 1, A) / r
    nn = torch.sqrt(nx * nx + ny * ny)
    d = (sgx * nx + sgy * ny) / nn
    keep = slid > lf.LID_MIN
    cnt = keep.sum(-1)
    score = torch.where(keep, d, torch.zeros_like(d)).sum(-1) / cnt.clamp(min=1)
    score = torch.where(cnt >= lf.MIN_ANGLES, score, torch.full_like(score, -1e9))
    best = score.reshape(-1).argmax()
    j1, j2, jb = np.unravel_index(int(best), (O1, O2, NB))
    circ = score[:, :, -1].reshape(-1).argmax()
    k1, k2 = np.unravel_index(int(circ), (O1, O2))
    return dict(frame=i, side=side, fitted=bool(e.get('fitted')), reason=e.get('reason'),
                c0=c0.tolist(), r=r,
                ell=dict(cx=float(c0[0] + offx[j2]), cy=float(c0[1] + offy[j1]), b=float(bs[jb]),
                         score=float(score[j1, j2, jb]), n=int(cnt[j1, j2, jb])),
                circ=dict(cx=float(c0[0] + offx[k2]), cy=float(c0[1] + offy[k1]),
                          score=float(score[k1, k2, -1]), n=int(cnt[k1, k2, -1])),
                here=float(score[O1 // 2, O2 // 2, -1]))


if __name__ == '__main__':
    items = [(int(a), b) for a, b in (x.split(':') for x in sys.argv[1].split(','))]
    out = [search(i, s) for i, s in items]
    for o in out:
        print('%4d %-5s fit=%-5s  circle@track %.4f | best circle %.4f d=(%+.1f,%+.1f) | '
              'ellipse b/r %.2f %.4f d=(%+.1f,%+.1f) gain %+.1f%%' % (
                  o['frame'], o['side'], o['fitted'], o['here'], o['circ']['score'],
                  o['circ']['cx'] - o['c0'][0], o['circ']['cy'] - o['c0'][1],
                  o['ell']['b'] / o['r'], o['ell']['score'], o['ell']['cx'] - o['c0'][0],
                  o['ell']['cy'] - o['c0'][1], 100 * (o['ell']['score'] / max(o['circ']['score'], 1e-9) - 1)))
    if len(sys.argv) > 2:
        json.dump(out, open(sys.argv[2], 'w'))
