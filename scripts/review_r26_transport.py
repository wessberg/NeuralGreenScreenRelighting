"""Show held observed-RGB comparisons; black is unscored, never a repaired pupil."""
import argparse
import cv2
import numpy as np
from color_pipeline import linear_to_srgb
from shot_paths import add_shot_argument, from_args


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    args=ap.parse_args(); shot=from_args(args)
    out=shot.composite/'r26_transport/review'; out.mkdir(parents=True,exist_ok=True)
    for mode in range(5):
        rows=[]
        label=f'donor source {(730,760,790,820)[mode]}' if mode<4 else 'equal mean of 3+ paired donors'
        for side in ('left','right'):
            with np.load(shot.relight/f'neural_iris_observations/000105_{side}.npz') as d:
                source=d['linear_rgb']
            with np.load(shot.composite/f'r26_joint/prediction_{side}.npz') as d:
                xy,pair,pred,control=[d[k] for k in ('xy','paired','predictions','control_predictions')]
            if mode<4: mask=pair[mode]; a=pred[mode]; b=control[mode]
            else:
                count=pair.sum(0); mask=count>=3
                a=pred.sum(0)/np.maximum(count[:,None],1)
                b=control.sum(0)/np.maximum(count[:,None],1)
            panels=[]
            for title,values in [('SOURCE CHECK',source[xy[:,1],xy[:,0]]),('JOINT PREDICTION',a),('CENTRE/SCALE',b)]:
                rgb=np.zeros_like(source); rgb[xy[mask,1],xy[mask,0]]=values[mask]
                display=np.clip(linear_to_srgb(rgb)[...,::-1]*255,0,255).astype('uint8')
                image=cv2.resize(display,(256,256),interpolation=cv2.INTER_NEAREST)
                header=np.zeros((32,256,3),'uint8')
                cv2.putText(header,title,(6,22),0,.5,(255,255,255),1)
                panels.append(np.concatenate([header,image],0))
            header=np.zeros((52,768,3),'uint8')
            cv2.putText(header,f'{side.upper()} target source 805 | {label} | {int(mask.sum())} pixels',(8,20),0,.5,(255,255,255),1)
            cv2.putText(header,'Black = outside paired check. Observed RGB sums, NOT clean iris.',(8,43),0,.48,(220,220,220),1)
            rows.append(np.concatenate([header,np.concatenate(panels,1)],0))
        header=np.zeros((42,768,3),'uint8')
        cv2.putText(header,'HELD TARGET / DONOR COMPARISONS - 1 sec each, NOT motion or repair',(8,27),0,.5,(255,255,255),1)
        board=np.concatenate([header,*rows],0)
        for j in range(24): assert cv2.imwrite(str(out/f'{mode*24+j:06d}.png'),board)
    print('R26_TRANSPORT_REVIEW_PREPARED')


if __name__=='__main__': main()
