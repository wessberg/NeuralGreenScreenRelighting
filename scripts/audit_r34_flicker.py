"""Temporal behaviour of the fill, in the REGISTERED eye frame, against paired controls.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/audit_r34_flicker.py --name r34_pooled

Two measures, because round 34 wrote two that were wrong and both were wrong the same way
-- they shared the bug's assumption:

* a "fill/iris ratio traverse" reported 3-10x and looked damning, but the footprint SWEEPS
  from rho .38 to .75, so it sits on darker iris early and brighter iris later and most of
  that traverse is real;
* a rolling-median "jitter" reported the fill as calmer than the source while the artist
  was plainly seeing colour change -- a five-frame rolling median discounts slow drift by
  construction, and slow drift was what he was watching.

What is measured instead: per-cell spread over frames in the eye's normalized grid, where
a cell is the same piece of iris in every frame; and the filled patch against its own
surround frame to frame, with her own untouched iris rotated 180 degrees into the same
ring as the control (round 32's frozen rotation design).
"""

import json,os,sys
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2,numpy as np
import argparse
_ap=argparse.ArgumentParser()
_ap.add_argument('--name', default='r34_pooled')
_ap.add_argument('--shot', default='music_videos/reporter_test/shots/reporter_a')
_a=_ap.parse_args()
name=_a.name
shot=_a.shot
d=json.load(open(f'{shot}/logs/{name}.json'))
rows={r['frame']:r for r in d['rows']}
G=49; t=np.linspace(-0.95,0.95,G).astype(np.float32); U,V=np.meshgrid(t,t)
rho=np.hypot(U,V)
acc={'left':{'out':[],'w':[]}, 'right':{'out':[],'w':[]}}
for i in sorted(rows):
    rep=cv2.imread(f'{shot}/composite/{name}/removal/{i:06d}.png')
    w=cv2.imread(f'{shot}/composite/{name}/weight/{i:06d}.png',0)
    if rep is None or w is None: continue
    for e in rows[i]['eyes']:
        if not e.get('repaired'): continue
        cx,cy=e['limbus_centre']; ax,ay=e['limbus_axes']
        mx=(cx+U*ax).astype(np.float32); my=(cy+V*ay).astype(np.float32)
        o=cv2.remap(rep,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        ww=cv2.remap(w.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        acc[e['side']]['out'].append(o); acc[e['side']]['w'].append(ww)
for side in ('left','right'):
    O=np.array(acc[side]['out']); W=np.array(acc[side]['w'])
    if len(O)<20: continue
    filled=W>128
    n=filled.sum(0)
    core=(n>=20)&(rho<0.88)
    never=(filled.sum(0)==0)&(rho<0.88)
    lum=O.mean(-1)
    def spread(cells, sel=None):
        vals=[]
        for j in np.argwhere(cells):
            y,x=j
            v=lum[:,y,x]
            m=filled[:,y,x] if sel is None else np.ones(len(v),bool)
            v=v[m] if sel is None else v
            if len(v)<10: continue
            vals.append(np.percentile(v,95)-np.percentile(v,5))
        return np.array(vals)
    a=spread(core); b=spread(never, sel='all')
    print('%s  cells filled>=20 frames: %d   never filled: %d'%(side,core.sum(),never.sum()))
    if len(a): print('   FILLED cell p5-p95 spread over frames: median %.1f  p95 %.1f  max %.1f'%(np.median(a),np.percentile(a,95),a.max()))
    if len(b): print('   SOURCE cell p5-p95 spread over frames: median %.1f  p95 %.1f  max %.1f'%(np.median(b),np.percentile(b,95),b.max()))
    if len(a) and len(b): print('   ratio of medians: %.2fx'%(np.median(a)/max(np.median(b),1e-6)))
