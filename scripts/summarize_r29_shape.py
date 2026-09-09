"""Measured attribution diagnostics on frozen R29 predictions; never refits."""
import argparse
import json
import numpy as np
from audit_r24_source import write
from audit_r27_photometry import error
from audit_r29_shape import prepare, training_values
from eye_shape_fit import design
from shot_paths import add_shot_argument, from_args


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap); shot=from_args(ap.parse_args())
    report=json.loads((shot.logs/'r29_shape.json').read_text())
    result={}
    for side,r in report['results'].items():
        p=prepare(shot,side,np.array(report['focal']))
        truth=training_values(shot,side,p,range(6))
        chosen=r['records'][r['selected']]['config'] if r['selected'] is not None else 0
        matched={a:next(x for x in r['records'] if x['config']==chosen and x['arm']==a)
                 for a in ('appearance','broad','narrow','spatial_backing')}
        detail={}; sampling=[]
        with np.load(shot.composite/f'r29_shape/predictions_{side}.npz') as z:
            for arm,rec in matched.items():
                i=rec['index']; c=rec['config']; m=rec['appearance_count']; frames=[]
                v,co=z[f'vectors_{i}'],z[f'co_{i}']
                for j in range(6):
                    x,_=design(p['trans'][j][c],v,arm=='spatial_backing')
                    layer=z[f'layers_{i}_{j}']; mask=p['masks'][j] if j>=4 else np.ones(len(x),bool)
                    label=mask&p['labels'][j]
                    backing=x[:,289:m]@co[289:m]
                    frames.append(dict(source_frame=p['obs'][j]['frame']+700,
                        support='candidate' if j<4 else 'joint',
                        all=error(layer.sum(0)[mask],truth[j][mask]),
                        reflection=error(layer.sum(0)[label],truth[j][label]),
                        labelled_mean_predicted_transmitted=float(layer[0,label].mean()) if label.any() else None,
                        labelled_mean_predicted_backing=float(backing[label].mean()) if label.any() else None,
                        labelled_mean_predicted_reflected=float(layer[1,label].mean()) if label.any() else None))
                detail[arm]=dict(index=i,geometry=rec['geometry'],box_n=rec['box_n'],
                    selection_score=rec['selection_score'],train_reflection=rec['train_reflection'],frames=frames)
            for rec in r['records']:
                if rec['box_n']!=3: continue
                i,c,arm=rec['index'],rec['config'],rec['arm']; co=z[f'co_{i}']; v=z[f'vectors_{i}']
                high=next(x for x in r['records'] if x['geometry']==rec['geometry'] and x['box_n']==7 and x['arm']==arm)
                differences=[]; refits=[]
                for j in range(6):
                    x,_=design(p['trans'][j][c+1],v,arm=='spatial_backing')
                    low=z[f'layers_{i}_{j}'].sum(0)
                    differences.append(float(np.max(np.abs(x@co-low))))
                    refits.append(float(np.max(np.abs(z[f'layers_{high["index"]}_{j}'].sum(0)-low))))
                sampling.append(dict(geometry=rec['geometry'],arm=arm,
                    fixed_coefficients_3_vs_7_max_linear=max(differences),refitted_3_vs_7_max_linear=max(refits)))
        result[side]=dict(selected=r['selected'],matched_config=chosen,
            fixed_witness_when_unselected=r['selected'] is None,matched=detail,sampling=sampling)
    write(shot.logs/'r29_attribution.json',result)
    for side,r in result.items():
        print(side,'matched config',r['matched_config'])
        for arm,d in r['matched'].items():
            print(arm,'train',d['train_reflection']['median_display'],'CV',d['selection_score'],
                  '815',d['frames'][-1]['reflection'])
        print('sampling fixed/refit maxima',max(x['fixed_coefficients_3_vs_7_max_linear'] for x in r['sampling']),
              max(x['refitted_3_vs_7_max_linear'] for x in r['sampling']))


if __name__=='__main__': main()
