"""Read-only fit replay, held RGB guards, protected fallbacks and decoded reviews."""
import argparse
import json
from unittest.mock import patch
import numpy as np
import audit_r29_shape as audit
from audit_r24_source import ROOT, sha, write
from audit_r27_photometry import error
from coupled_eye_relight import apply_coupled
from verify_r29_fixture import controls
from shot_paths import add_shot_argument, from_args


def replay(shot):
    report=json.loads((shot.logs/'r29_shape.json').read_text())
    assert audit.freeze(shot)==report['evidence']
    for name,value in report['code_sha256'].items(): assert sha(ROOT/'scripts'/name)==value
    for path,value in report['artifact_hashes'].items(): assert sha(ROOT/path)==value
    results={}; out=shot.composite/'r29_shape'
    for side in ('left','right'):
        saved=audit.read_rgb; denied=[]
        def guard(s,which,frame):
            if frame in (105,115): denied.append(frame); raise AssertionError('Seen diagnostic RGB entered fit')
            return saved(s,which,frame)
        with patch.object(audit,'read_rgb',guard):
            p=audit.prepare(shot,side,np.array(report['focal']))
            truth=audit.training_values(shot,side,p,range(4))
            result=report['results'][side]
            assert {str(k):v for k,v in p['support'].items()}==result['support']
            models=folds=0
            with np.load(out/f'predictions_{side}.npz') as z:
                for j in range(6):
                    assert np.array_equal(z[f'xy_{j}'],p['xy'][j])
                    assert np.array_equal(z[f'mask_{j}'],p['masks'][j])
                    assert np.array_equal(z[f'label_{j}'],p['labels'][j])
                for r in result['records']:
                    i,c,arm=r['index'],r['config'],r['arm']
                    xs,m,v,co=audit.model(p,c,arm,list(range(4)),truth)
                    assert np.array_equal(co,z[f'co_{i}']) and np.array_equal(v,z[f'vectors_{i}'])
                    for j in range(6):
                        layer=np.stack([xs[j][:,:m]@co[:m],xs[j][:,m:]@co[m:]])
                        assert np.array_equal(layer,z[f'layers_{i}_{j}'])
                    models+=1
                    for held in range(4):
                        train=[j for j in range(4) if j!=held]
                        class GuardedTruth(dict):
                            def __getitem__(self,k):
                                assert k in train, 'Held fold RGB accessed by fitter'
                                return super().__getitem__(k)
                        xf,mf,vf,cf=audit.model(p,c,arm,train,GuardedTruth(truth))
                        pred=xf[held]@cf; mask=p['masks'][held]; label=p['labels'][held]
                        actual=r['folds'][held]
                        assert error(pred[mask&label],truth[held][mask&label])==actual['reflection']
                        assert error(pred[mask&~label],truth[held][mask&~label])==actual['other']
                        if held in result['eligible_folds']:
                            balanced=float((np.mean((pred[mask&label]-truth[held][mask&label])**2)
                                +np.mean((pred[mask&~label]-truth[held][mask&~label])**2))/2)
                            assert balanced==actual['balanced_mse']
                        else: assert actual['balanced_mse'] is None
                        folds+=1
                    expected=float(np.mean([r['folds'][j]['balanced_mse'] for j in result['eligible_folds']])) if len(result['eligible_folds'])>=2 else None
                    assert r['selection_score']==expected
                    print('R29_REPLAY',side,i,flush=True)
        assert not denied
        assert {str(k):v for k,v in audit.score(shot,side,p,out,result).items()}==result['seen_scores']
        ranks=sorted(result['records'],key=lambda r:(r['selection_score'],r['index'])) if len(result['eligible_folds'])>=2 else []
        assert result['selected']==(ranks[0]['index'] if ranks else None)
        transactions=0
        for o in p['obs']:
            src=saved(shot,side,o['frame'])
            output,info=apply_coupled(src,np.ones(src.shape[:2],bool),object(),object(),geometry_reliable=False)
            assert np.array_equal(output,src) and info['changed_channels']==0
            transactions+=1
        results[side]=dict(full_models_replayed=models,held_folds_replayed=folds,
            held_fold_rgb_access_guarded=True,seen_805_815_fit_reads=0,disabled_transactions=transactions)
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--skip-videos',action='store_true')
    ap.add_argument('--videos-only',action='store_true'); args=ap.parse_args(); shot=from_args(args)
    from verify_r15 import fallback
    if args.videos_only:
        result=json.loads((shot.logs/'r29_verification.json').read_text())
        report=json.loads((shot.logs/'r29_shape.json').read_text())
        assert audit.freeze(shot)==report['evidence']
        for name,value in report['code_sha256'].items(): assert sha(ROOT/'scripts'/name)==value
        for path,value in report['artifact_hashes'].items(): assert sha(ROOT/path)==value
        result['fixture']=controls()
        result['fallback']=fallback(shot)
    else:
        result=dict(fixture=controls(),replay=replay(shot),fallback=fallback(shot))
    if not args.skip_videos:
        from verify_r20_eyes import video_checks
        result['videos']={name:video_checks(shot,name,count,fps,shape) for name,count,fps,shape in
            [('r29_shape',150,float(shot.frame_rate),(960,768,3)),('r29_layers',144,24.,(580,1000,3))]}
        import cv2
        for name,frames in [('r29_shape',(30,60,90,105,115,120)),('r29_layers',(0,24,48,72,96,120))]:
            cap=cv2.VideoCapture(str(shot.final/f'{name}.mp4'))
            for f in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,im=cap.read(); assert ok
                assert cv2.imwrite(str(shot.composite/f'{name}/decoded_{f:03d}.png'),im)
            cap.release()
    result['verifier_sha256']=sha(ROOT/'scripts/verify_r29_shape.py')
    write(shot.logs/'r29_verification.json',result)
    print('R29_VERIFIED',json.dumps(result['fallback']),flush=True)


if __name__=='__main__': main()
