"""Is the tracked limbus actually ON her iris? Measured against the IMAGE.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/audit_r34_track.py --name r34_newiris2

Round 33 measured this landmark's jitter at 0.353 px rms against a local quadratic fit of
its own trajectory and concluded it was solid. That measured SMOOTHNESS, not ACCURACY: a
smoothly wrong anchor is still wrong, and everything in rounds 32-34 is anchored on it.

This walks outward from the tracked ellipse at nasal and temporal angles -- clear of the
lids, where sclera is always visible -- and finds the iris/sclera step in the photograph
itself. Round 34 result: observed radius over tracked radius is 0.936 median on the left
eye (a 13 percent systematic size error on a 12.9 px iris), sd 1.66 px, and p95
frame-to-frame change 2.49 px. That is the artist's "double iris" and his "delayed, slow
to react", and it is upstream of every fill this project has tried.
"""

import json,os,sys
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2,numpy as np
import argparse
_ap=argparse.ArgumentParser()
_ap.add_argument('--name', default='r34_newiris2')
_ap.add_argument('--shot', default='music_videos/reporter_test/shots/reporter_a')
_a=_ap.parse_args()
name=_a.name
shot=_a.shot
d=json.load(open(f'{shot}/logs/{name}.json'))
rows={r['frame']:r for r in d['rows']}
TH=np.concatenate([np.linspace(-0.5,0.5,9), np.linspace(np.pi-0.5,np.pi+0.5,9)])  # nasal/temporal
RR=np.linspace(0.6,1.5,60)
res={'left':[], 'right':[]}
for i in sorted(rows):
    src=cv2.imread(f'{shot}/composite/{name}/source/{i:06d}.png')
    if src is None: continue
    lum=src.max(-1).astype(np.float32)
    for e in rows[i]['eyes']:
        if not e.get('repaired'): continue
        cx,cy=e['limbus_centre']; ax,ay=e['limbus_axes']
        edges=[]
        for t in TH:
            xs=(cx+RR*ax*np.cos(t)).astype(np.float32)[None,:]
            ys=(cy+RR*ay*np.sin(t)).astype(np.float32)[None,:]
            prof=cv2.remap(lum,xs,ys,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]
            g=np.gradient(prof)
            k=int(np.argmax(g))
            if g[k]<=2: continue
            edges.append(RR[k])
        if len(edges)>=8:
            res[e['side']].append((i,float(np.median(edges)),float(np.std(edges))))
for side in ('left','right'):
    v=res[side]
    if len(v)<20: continue
    r=np.array([x[1] for x in v]); idx=np.array([x[0] for x in v])
    adj=np.diff(idx)==1
    step=np.abs(np.diff(r))[adj]
    print('%s n=%d'%(side,len(v)))
    print('   observed limbus radius / tracked radius: median %.3f  p5 %.3f  p95 %.3f  sd %.3f'%(
        np.median(r),np.percentile(r,5),np.percentile(r,95),r.std()))
    print('   frame-to-frame change of that ratio: median %.4f  p95 %.4f  max %.4f'%(
        np.median(step),np.percentile(step,95),step.max()))
    ax=np.array([rows[i]['eyes'][0]['limbus_axes'][0] for i,_,_ in v])
    print('   in native pixels: sd %.2f px   p95 frame-to-frame %.2f px'%(
        (r.std()*np.median(ax)),(np.percentile(step,95)*np.median(ax))))
