"""Sequential preview and objective matte/detail/lighting checks before full export."""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
from pathlib import Path
import json, subprocess, numpy as np, cv2
from composite import compose_frame,load_exr,FFMPEG
from color_pipeline import sdr_preview,srgb_to_linear
ROOT=Path(__file__).resolve().parents[1]
import sys as _sys; _sys.path.insert(0,str(ROOT/'scripts'))
from shot_paths import add_shot_argument, from_args
cv2.setNumThreads(8)
bg=cv2.resize(load_exr(ENV/'test_background_shadow.exr')[...,:3][...,::-1],(1280,720),interpolation=cv2.INTER_LANCZOS4)
log=open(SHOT/'logs/preview_encode.log','w')
cmd=[str(FFMPEG),'-hide_banner','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','pipe:0','-i',str(SHOT/'source/plate.mp4'),'-map','0:v:0','-map','1:a:0','-c:v','h264_nvenc','-preset','p6','-cq','18','-b:v','0','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-t','10','-movflags','+faststart',str(SHOT/'composite/preview_sdr.mp4')]
proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=log)
report=[];sequences={}
for i in range(300):
    comp,a=compose_frame(i,bg,(1280,720))
    view=(sdr_preview(comp)*255).astype(np.uint8)
    proc.stdin.write(view.tobytes())
    for start in [0,50,140,230,288]:
        if start<=i<start+12:
            tile=cv2.resize(view,(320,180));cv2.putText(tile,f'{i:03d}',(8,20),0,.55,(255,255,255),1)
            sequences.setdefault(start,[]).append(tile)
    alpha=load_exr(SHOT/f'key/alpha/{i:06d}.exr')
    source=cv2.imread(str(SHOT/f'source/frames/{i:06d}.png'),cv2.IMREAD_UNCHANGED)[...,::-1]/65535
    screen=(source[...,1]>.68)&(source[...,1]-np.maximum(source[...,0],source[...,2])>.48)
    foreground=(~screen).astype(np.uint8)
    core=cv2.erode(foreground,np.ones((13,13),np.uint8))>0
    fg=load_exr(SHOT/f'key/fg/{i:06d}.exr')[...,::-1]
    err=np.abs(fg-srgb_to_linear(source))[core]
    holes=(alpha<.5)&core
    count,_,stats,_=cv2.connectedComponentsWithStats(holes.astype(np.uint8))
    largest_hole=int(stats[1:,cv2.CC_STAT_AREA].max()) if count>1 else 0
    gain=np.load(SHOT/f'relight/controls/{i:06d}.npz')['gain'].astype(np.float32)
    mask=cv2.resize(alpha,gain.shape[1::-1],interpolation=cv2.INTER_AREA)>.99
    report.append({'frame':i,'core_alpha_mean':float(alpha[core].mean()),'core_pixels_below_half':int(holes.sum()),'largest_core_hole_pixels':largest_hole,'source_detail_linear_mae':float(err.mean()),'source_detail_max_error':float(err.max()),'mean_rgb_gain':gain[mask].mean(0).tolist()})
    if i%30==0:print('PREVIEW_QC',i,flush=True)
proc.stdin.close();rc=proc.wait();log.close()
if rc:raise RuntimeError('Preview encoder failed')
for start,imgs in sequences.items():
    while len(imgs)<12:imgs.append(imgs[-1])
    sheet=np.vstack([np.hstack(imgs[j:j+4]) for j in range(0,12,4)])
    cv2.imwrite(str(SHOT/f'composite/sequence_{start:03d}.jpg'),sheet[...,::-1])
g=np.array([r['mean_rgb_gain'] for r in report]);delta=np.max(np.abs(np.diff(g,axis=0)),axis=1)
summary={'frames':len(report),'minimum_core_alpha_mean':min(r['core_alpha_mean'] for r in report),'largest_core_hole_pixels':max(r['largest_core_hole_pixels'] for r in report),'max_core_source_detail_mae':max(r['source_detail_linear_mae'] for r in report),'max_frame_mean_gain_step':float(delta.max()),'p95_frame_mean_gain_step':float(np.percentile(delta,95)),'note':'Mean changes include real actor motion. No spatial actor warp or generated image pixels are used in relighting. Sequential images require visual review.','per_frame':report}
(SHOT/'logs/preview_qc.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='per_frame'},indent=2))
