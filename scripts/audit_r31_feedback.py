"""Match artist eye screenshots to saved trial frames without modifying the videos."""
import argparse
import json
from pathlib import Path
from diffusion_eye_trial import write, sha
from shot_paths import add_shot_argument, from_args
import cv2
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--screenshots',nargs='+',required=True)
    args=ap.parse_args(); shot=from_args(args); root=shot.composite/'r31_reflection_spots'
    decoded=[]; cap=cv2.VideoCapture(str(shot.final/'r31_reflection_spots_slow.mp4'))
    while True:
        ok,frame=cap.read()
        if not ok: break
        decoded.append(frame[426:586,384:768])
    cap.release(); results=[]
    for path in args.screenshots:
        im=cv2.imread(path)
        if im is None: raise FileNotFoundError(path)
        small=cv2.resize(im,(320,round(320*im.shape[0]/im.shape[1])),interpolation=cv2.INTER_AREA)
        template=small[12:-12,20:-20]
        rankings=[]
        for i,frame in enumerate(decoded):
            best=(-1,None)
            for width in range(285,421,3):
                candidate=cv2.resize(frame,(width,round(width*frame.shape[0]/frame.shape[1])),interpolation=cv2.INTER_LINEAR)
                if min(np.array(candidate.shape[:2])-template.shape[:2])<0: continue
                corr=cv2.matchTemplate(candidate,template,cv2.TM_CCOEFF_NORMED)
                _,score,_,xy=cv2.minMaxLoc(corr)
                if score>best[0]: best=(score,dict(width=width,xy=xy))
            rankings.append(dict(local_frame=i,source_frame=700+i,correlation=best[0],transform=best[1]))
        rankings.sort(key=lambda x:x['correlation'],reverse=True)
        results.append(dict(screenshot=str(Path(path)),sha256=sha(path),best_matches=rankings[:5],
                            matching_is_inference_not_frame_metadata=True))
    record=dict(results=results,video_sha256=sha(shot.final/'r31_reflection_spots_slow.mp4'),
                video_not_modified=True)
    write(shot.logs/'r31_feedback_matching.json',record); print(json.dumps(record,indent=2))


if __name__=='__main__': main()
