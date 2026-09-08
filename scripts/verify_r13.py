"""Verify neural adapter conventions, hidden-input exclusion, fallback and review media."""
import argparse
import ast
import hashlib
import json
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
import torch
from anamorphic import SQUEEZE
from ellseg_iris import EllSeg, prepare, ROOT
from ritnet_iris import RITnet
from shot_paths import add_shot_argument,from_args
from temporal_iris import apply_eye_repair


def main():
    ap=argparse.ArgumentParser(description=__doc__);add_shot_argument(ap)
    args=ap.parse_args();shot=from_args(args)
    out=shot.composite/'r13_ellseg'
    em,rm=EllSeg(),RITnet()
    check={}
    for name,model in [('EllSeg',em),('RITnet',rm)]:
        repo=ROOT/'dependencies'/name
        rev=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        assert rev==model.provenance['expected_revision']
        assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only'],text=True).strip()
    check['unmodified_author_revisions']=True
    # Compare preprocessing against the actual upstream function on a supported
    # rectangular input. Its square-input Python-3 slicing bug is deliberately fixed.
    code=ast.parse((ROOT/'dependencies/EllSeg/evaluate_ellseg.py').read_text())
    fn=next(n for n in code.body if isinstance(n,ast.FunctionDef) and n.name=='preprocess_frame')
    namespace={'cv2':cv2,'np':np,'torch':torch,'sys':sys}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<upstream preprocess>','exec'),namespace)
    native=np.load(shot.relight/'neural_iris_observations/000000_left.npz')['linear_rgb']
    from color_pipeline import linear_to_srgb
    rgb=linear_to_srgb(native)*255
    rectangular=cv2.resize(rgb,(80,60),interpolation=cv2.INTER_LINEAR)
    gray=cv2.cvtColor(np.clip(rectangular,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
    official,_=namespace['preprocess_frame'](gray,(240,320))
    adapted,_,_=prepare(rectangular)
    diff=float(np.max(np.abs(official.numpy()[0]-adapted)))
    assert diff<1e-6,diff
    check['upstream_preprocessing_max_abs']=diff
    # Corroborate ellipse angle/sign and affine mapping against the author's conic
    # equation, using several angles and both square/anamorphic native coordinates.
    from helperfunctions import my_ellipse
    conic_errors=[]
    for aspect in (1.,SQUEEZE):
        pred=em.predict(rgb,pixel_aspect=aspect)
        matrix=pred['native_to_model']
        for ellipse in pred['ellipses'].values():
            norm=ellipse['normalized_parameters'];curve=ellipse['boundary_xy']
            model=np.column_stack([curve,np.ones(len(curve))])@matrix.T
            pts=np.column_stack([model[:,:2]/[160,120]-1,np.ones(len(curve))])
            residual=np.abs(np.einsum('ni,ij,nj->n',pts,my_ellipse(norm).mat,pts)).max()
            assert residual<1e-4,residual
            conic_errors.append(float(residual))
        for angle in (-1.1,-.3,.45,1.2):
            p=np.array([.2,-.1,.6,.3,angle]);t=np.linspace(0,2*np.pi,90)
            cs,sn=np.cos(angle),np.sin(angle)
            curve=np.column_stack([p[2]*np.cos(t),p[3]*np.sin(t)])@np.array([[cs,sn],[-sn,cs]])+p[:2]
            pts=np.column_stack([curve,np.ones(len(curve))])
            assert np.abs(np.einsum('ni,ij,nj->n',pts,my_ellipse(p).mat,pts)).max()<1e-10
    check['author_conic_max_residual']=max(conic_errors)
    # Scrub before every learned operation; test real RGB with two extreme hidden
    # substitutions. This does not certify the upstream face tracker or donors.
    yy,xx=np.mgrid[:rgb.shape[0],:rgb.shape[1]]
    hidden=(xx-rgb.shape[1]//2)**2+(yy-rgb.shape[0]//2)**2<=4
    invariance=[]
    for aspect in (1.,SQUEEZE):
        a=em.predict(rgb,pixel_aspect=aspect,excluded=hidden)
        ra=rm.predict(rgb,pixel_aspect=aspect,excluded=hidden)
        for value in ([255,0,255],[0,255,0]):
            altered=rgb.copy();altered[hidden]=value
            b=em.predict(altered,pixel_aspect=aspect,excluded=hidden)
            rb=rm.predict(altered,pixel_aspect=aspect,excluded=hidden)
            assert np.array_equal(a['labels'],b['labels']) and np.array_equal(ra,rb)
            for name in a['ellipses']:
                assert np.array_equal(a['ellipses'][name]['boundary_xy'],b['ellipses'][name]['boundary_xy'])
            invariance.append({'pixel_aspect':aspect,'hidden_rgb':value,'exact':True})
    check['hidden_rgb_invariance']=invariance
    check['constant_input_abstention']=all((rm.predict(np.full((64,64,3),v,np.float32))==-1).all()
        and not em.predict(np.full((64,64,3),v,np.float32))['ellipses'] for v in (0,128,255))
    assert check['constant_input_abstention']
    # Validate exact keyed crop persistence and fallback for every saved observation.
    manifest=json.loads((shot.relight/'neural_iris_observations/manifest.json').read_text())
    n=0
    for row in manifest['frames']:
        original=cv2.imread(str(shot.fg(row['frame'])),cv2.IMREAD_UNCHANGED)[...,::-1]
        for side,eye in row['eyes'].items():
            assert not eye['scene_reflection_allowed'] and not eye['geometry_validated'] and not eye['visibility_validated']
            if 'origin_xy' not in eye:continue
            data=np.load(shot.relight/'neural_iris_observations'/('%06d_%s.npz'%(row['frame'],side)))
            source=data['linear_rgb'];h,w=source.shape[:2];x,y=eye['origin_xy']
            assert np.array_equal(source,original[y:y+h,x:x+w])
            fallback,info=apply_eye_repair(source,np.ones((h,w),bool),geometry_reliable=False)
            assert np.array_equal(fallback,source) and info['changed_channels']==0
            assert data['visible_part_proposal'].shape==(h,w)
            n+=1
    check['exact_keyed_eye_fallbacks']=n
    old=json.loads((shot.composite/'r10_iris/correspondence_detail.json').read_text())
    protected=old['checks']['protected_manifest']
    changed=[p for p,h in protected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    check['protected_files_unchanged']=len(protected)
    videos=[]
    for filename,expected in [('r13_ellseg.mp4',1200),('r13_ritnet.mp4',shot.frame_count),
                              ('r13_neural_observations.mp4',shot.frame_count)]:
        path=shot.final/filename
        if not path.exists():raise FileNotFoundError(path)
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);n=0;thumbs=[];sheets=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            # Every frame, ordered, aspect-preserving; sheets show temporal ordering,
            # with separate native overlays required for boundary inspection.
            thumb=cv2.resize(frame,(128,round(128*frame.shape[0]/frame.shape[1])))
            cv2.putText(thumb,str(n),(2,12),0,.32,(255,255,255),1);thumbs.append(thumb);n+=1
            if len(thumbs)==100:
                sheet=np.concatenate([np.concatenate(thumbs[j:j+10],1) for j in range(0,100,10)],0)
                target=out/(path.stem+'_sheet_%04d.jpg'%n)
                assert cv2.imwrite(str(target),sheet);sheets.append(str(target.relative_to(shot.dir)));thumbs=[]
        cap.release()
        if thumbs:
            while len(thumbs)%10:thumbs.append(np.zeros_like(thumbs[0]))
            sheet=np.concatenate([np.concatenate(thumbs[j:j+10],1) for j in range(0,len(thumbs),10)],0)
            target=out/(path.stem+'_sheet_%04d.jpg'%n)
            assert cv2.imwrite(str(target),sheet);sheets.append(str(target.relative_to(shot.dir)))
        assert n==expected and abs(fps-float(shot.frame_rate))<.001,(filename,n,fps)
        videos.append({'file':filename,'decoded_frames':n,'fps':fps,'sheets':sheets,
                       'visual_inspection':'pending','artist_approval':False})
    check['videos']=videos
    (shot.logs/'r13_verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in check.items() if k!='videos'},indent=2))


if __name__=='__main__':main()
