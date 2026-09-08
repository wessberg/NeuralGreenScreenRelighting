"""Official CorridorKey API, original-core preservation, separate float alpha/FG."""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
os.environ['CORRIDORKEY_SKIP_COMPILE']='1'
from pathlib import Path
import sys, json, time, argparse, logging
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'dependencies/CorridorKey'))
import numpy as np, cv2, torch
from CorridorKeyModule import CorridorKeyEngine
logging.basicConfig(level=logging.INFO)
cv2.setNumThreads(8)
torch.set_num_threads(8)

def linear(x):
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def hint_for(rgb):
    # Plate contains green wardrobe. Only its saturated, bright artificial screen is background.
    r,g,b=np.moveaxis(rgb,-1,0)
    bg=(g>.68)&(g-np.maximum(r,b)>.48)
    hint=1-bg.astype(np.float32)
    hint=cv2.erode(hint,np.ones((3,3),np.uint8))
    return cv2.GaussianBlur(hint,(0,0),1.0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');args=p.parse_args()
    for d in ['alpha','fg','qc']:(ROOT/'key'/d).mkdir(exist_ok=True)
    engine=CorridorKeyEngine(str(ROOT/'dependencies/CorridorKey/CorridorKeyModule/checkpoints/CorridorKey_v1.0.safetensors'),device='cuda',img_size=2048)
    ids=[0,60,150,240,299] if args.test else range(300)
    report=[];start=time.time()
    for i in ids:
        out=ROOT/f'key/fg/{i:06d}.exr'
        if out.exists() and not args.test:continue
        bgr=cv2.imread(str(ROOT/f'source/frames/{i:06d}.png'),cv2.IMREAD_UNCHANGED)
        rgb=np.ascontiguousarray(bgr[...,::-1].astype(np.float32)/65535)
        hint=hint_for(rgb)
        with torch.inference_mode():
            result=engine.process_frame(rgb,hint,input_is_linear=False,despill_strength=0,auto_despeckle=False,generate_comp=False,post_process_on_gpu=False)
        alpha=np.clip(result['alpha'][...,0],0,1)
        # Remove model specks only in the unquestionably clean screen far outside the silhouette.
        support=cv2.dilate((hint>.05).astype(np.uint8),np.ones((17,17),np.uint8))
        alpha*=support
        # Use original photographic pixels in the opaque body, preserving every captured detail.
        core=np.clip((alpha-.88)/.10,0,1)[...,None]
        fg=np.clip(result['fg'],0,1)*(1-core)+rgb*core
        # Protect green fabric: only despill antialias/hair pixels at the outside edge.
        edge=(1-core[...,0])*np.clip((fg[...,1]-np.maximum(fg[...,0],fg[...,2])-.08)*2,0,.8)
        fg[...,1]-=edge*(fg[...,1]-.5*(fg[...,0]+fg[...,2]))
        fg=linear(np.clip(fg,0,1)).astype(np.float32)
        flags=[cv2.IMWRITE_EXR_TYPE,cv2.IMWRITE_EXR_TYPE_HALF,cv2.IMWRITE_EXR_COMPRESSION,cv2.IMWRITE_EXR_COMPRESSION_ZIP]
        cv2.imwrite(str(ROOT/f'key/alpha/{i:06d}.exr'),alpha,flags)
        cv2.imwrite(str(out),fg[...,::-1],flags)
        interior=cv2.erode((hint>.99).astype(np.uint8),np.ones((9,9),np.uint8))>0
        stats={'frame':i,'alpha_mean_interior':float(alpha[interior].mean()),'interior_below_0.9':int((alpha[interior]<.9).sum()),'seconds':round(time.time()-start,2)}
        report.append(stats)
        if args.test:
            yy,xx=np.indices(alpha.shape);checker=np.where((xx//48+yy//48)%2,.22,.06)[...,None]
            c=fg*alpha[...,None]+checker*(1-alpha[...,None]);c=np.clip(np.where(c<=.0031308,c*12.92,1.055*c**(1/2.4)-.055),0,1)
            cv2.imwrite(str(ROOT/f'key/qc/{i:06d}_checker.jpg'),(c[...,::-1]*255).astype(np.uint8))
            cv2.imwrite(str(ROOT/f'key/qc/{i:06d}_alpha.png'),(alpha*65535).astype(np.uint16))
        if i%10==0 or args.test:print(json.dumps(stats),flush=True)
    (ROOT/('logs/key_test.json' if args.test else 'logs/key_run.json')).write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
