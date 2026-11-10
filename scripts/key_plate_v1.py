"""Official CorridorKey API, original-core preservation, separate float alpha/FG.

Shot-agnostic: pass --shot to key any shot; omit it for the proof-of-concept example at the
project root. Screen-detection thresholds and despill behaviour come from the shot's
shot.json ("keying"), because backing brightness and saturation, and whether the subject
wears green, differ between clips. The proof-of-concept plate has a green costume and so
must restrict despill to silhouette edges; a plate with non-green wardrobe does not.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
os.environ['CORRIDORKEY_SKIP_COMPILE']='1'
from pathlib import Path
import sys, json, time, argparse, logging
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'dependencies/CorridorKey'))
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np, cv2, torch
from CorridorKeyModule import CorridorKeyEngine
from shot_paths import Shot, add_shot_argument, from_args
logging.basicConfig(level=logging.INFO)
cv2.setNumThreads(8)
torch.set_num_threads(8)

def linear(x):
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def hint_for(rgb,cfg):
    # Only the saturated, bright artificial screen is background. On a plate with green
    # wardrobe these thresholds are what separate the costume from the backing.
    r,g,b=np.moveaxis(rgb,-1,0)
    bg=(g>cfg['green_min'])&(g-np.maximum(r,b)>cfg['green_dominance_min'])
    hint=1-bg.astype(np.float32)
    hint=cv2.erode(hint,np.ones((3,3),np.uint8))
    return cv2.GaussianBlur(hint,(0,0),1.0)

def test_frames(n):
    """Five representative frames spread across the selected range."""
    return sorted({0,n//5,2*n//5,3*n//5,n-1}) if n>=5 else list(range(n))

def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--test',action='store_true')
    add_shot_argument(p)
    args=p.parse_args()
    shot=from_args(args)
    for d in (shot.key_alpha,shot.key_fg,shot.key_qc,shot.logs):d.mkdir(parents=True,exist_ok=True)
    keying=shot.config.get('keying') or {}
    hint_cfg=shot.key_hint
    despill_mode=keying.get('despill_mode','edge_only')
    # CorridorKey's own despill defaults to 1.0; the proof of concept disables it (0.0) only
    # because that plate has a green costume the model would eat. A plate without green
    # wardrobe should use the engine's despill, which is what it is for.
    engine_despill=float(keying.get('engine_despill_strength',0))
    despill_offset_abs=float(keying.get('despill_offset_abs',0.08))
    despill_offset_rel=float(keying.get('despill_offset_rel',0.0))
    # NOTE: CorridorKey's despill_strength does NOT affect result['fg'], which is documented in
    # inference_engine.py as "sRGB, Raw Prediction (Straight)". Only result['processed'] is
    # despilled, and that is linear, premultiplied and garbage-matted, so this pipeline cannot
    # use it without losing the original-core blend. ALL despill here is the manual pass below.
    # The GATE and the LIMIT are separate choices, and shipping max(R,B) as the gate while
    # correcting to the average is what let spill through on dark hair. 'max' keeps that
    # behaviour for every existing shot; 'average' is the standard green-limit gate.
    despill_limit=keying.get('despill_limit','max')
    despill_gain=float(keying.get('despill_gain',2.0))
    despill_max=float(keying.get('despill_max',0.8))
    total=shot.frame_count
    print(json.dumps({'shot':shot.name,'frames':total,'hint':hint_cfg,'despill_mode':despill_mode,'despill_limit':despill_limit,
                      'engine_despill_strength':engine_despill,'despill_offset_abs':despill_offset_abs,
                      'despill_offset_rel':despill_offset_rel}),flush=True)
    engine=CorridorKeyEngine(str(ROOT/'dependencies/CorridorKey/CorridorKeyModule/checkpoints/CorridorKey_v1.0.safetensors'),device='cuda',img_size=2048)
    ids=test_frames(total) if args.test else range(total)
    report=[];start=time.time()
    for i in ids:
        out=shot.fg(i)
        if out.exists() and not args.test:continue
        bgr=cv2.imread(str(shot.frame(i)),cv2.IMREAD_UNCHANGED)
        if bgr is None:raise SystemExit(f'Missing source frame {shot.frame(i)}')
        rgb=np.ascontiguousarray(bgr[...,::-1].astype(np.float32)/65535)
        hint=hint_for(rgb,hint_cfg)
        with torch.inference_mode():
            result=engine.process_frame(rgb,hint,input_is_linear=False,despill_strength=engine_despill,auto_despeckle=False,generate_comp=False,post_process_on_gpu=False)
        alpha=np.clip(result['alpha'][...,0],0,1)
        # Remove model specks only in the unquestionably clean screen far outside the silhouette.
        support=cv2.dilate((hint>.05).astype(np.uint8),np.ones((17,17),np.uint8))
        alpha*=support
        # Use original photographic pixels in the opaque body, preserving every captured detail.
        core=np.clip((alpha-.88)/.10,0,1)[...,None]
        fg=np.clip(result['fg'],0,1)*(1-core)+rgb*core
        if despill_mode!='off':
            # Green-excess removal. edge_only protects green wardrobe by restricting the
            # correction to antialias/hair pixels outside the opaque core.
            # The offset must scale with brightness. A purely absolute threshold spares dim
            # partial-alpha rim pixels, where a cast of ~0.1x luminance is plainly visible but
            # measures well under it: on the reporter plate 20% of edge pixels stayed
            # green-positive at a median 0.019 excess against a 0.08 threshold.
            # offset_rel=0 with offset_abs=0.08 reproduces the proof-of-concept exactly.
            luma=fg@np.array([.2126,.7152,.0722],np.float32)
            offset=despill_offset_abs+despill_offset_rel*luma
            # weight 1.0 == the "average" limit (G driven to the mean of R and B), the same
            # operation the library performs. The legacy gain of 2 only reaches a meaningful
            # weight for excesses around 0.5; subtle rim spill of ~0.018 got a weight of 0.014,
            # i.e. 1.4% of the correction it needed.
            # Green only has to beat the AVERAGE of red and blue to read as a cast. Her dark
            # brown hair sits at R .141 G .113 B .070: green is 8% above (R+B)/2 and plainly
            # olive, while never coming close to red. Measured around her ear on frame 79,
            # 0.0% of subject pixels tripped the max(R,B) gate and 11.8% were green-leaning,
            # so the despill did nothing at all exactly where the artist could see it.
            # Gating on the average cannot overcorrect: a neutral pixel has G == (R+B)/2 and
            # is a fixed point, skin is red-dominant and stays negative, and the correction
            # below can never drive green past that same average.
            ref=(np.maximum(fg[...,0],fg[...,2]) if despill_limit=='max'
                 else .5*(fg[...,0]+fg[...,2]))
            excess=np.clip((fg[...,1]-ref-offset)*despill_gain,0,despill_max)
            weight=(1-core[...,0])*excess if despill_mode=='edge_only' else excess
            fg[...,1]-=weight*(fg[...,1]-.5*(fg[...,0]+fg[...,2]))
        fg=linear(np.clip(fg,0,1)).astype(np.float32)
        flags=[cv2.IMWRITE_EXR_TYPE,cv2.IMWRITE_EXR_TYPE_HALF,cv2.IMWRITE_EXR_COMPRESSION,cv2.IMWRITE_EXR_COMPRESSION_ZIP]
        cv2.imwrite(str(shot.alpha(i)),alpha,flags)
        cv2.imwrite(str(out),fg[...,::-1],flags)
        interior=cv2.erode((hint>.99).astype(np.uint8),np.ones((9,9),np.uint8))>0
        stats={'frame':i,'alpha_mean_interior':float(alpha[interior].mean()),'interior_below_0.9':int((alpha[interior]<.9).sum()),'seconds':round(time.time()-start,2)}
        report.append(stats)
        if args.test:
            yy,xx=np.indices(alpha.shape);checker=np.where((xx//48+yy//48)%2,.22,.06)[...,None]
            c=fg*alpha[...,None]+checker*(1-alpha[...,None]);c=np.clip(np.where(c<=.0031308,c*12.92,1.055*c**(1/2.4)-.055),0,1)
            cv2.imwrite(str(shot.key_qc/f'{i:06d}_checker.jpg'),(c[...,::-1]*255).astype(np.uint8))
            cv2.imwrite(str(shot.key_qc/f'{i:06d}_alpha.png'),(alpha*65535).astype(np.uint16))
        if i%10==0 or args.test:print(json.dumps(stats),flush=True)
    shot.log('key_test.json' if args.test else 'key_run.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
