"""Replay source fits, exclude target RGB, verify fallback and encoded reviews."""
import argparse
import json
from unittest.mock import patch
import numpy as np
from audit_r24_source import ROOT, sha, write
import audit_r28_radiance as audit
from coupled_eye_relight import apply_coupled
from verify_r28_fixture import controls
from shot_paths import add_shot_argument, from_args


def replay(shot):
    report=json.loads((shot.logs/'r28_radiance.json').read_text())
    assert audit.freeze(shot)==report['evidence']
    for name,value in report['code_sha256'].items(): assert sha(ROOT/'scripts'/name)==value
    out=shot.composite/'r28_radiance'; results={}
    for side in ('left','right'):
        r=report['results'][side]
        for path,value in r['artifact_hashes'].items(): assert sha(ROOT/path)==value
        obs,matrix,xy,directions,reflections,pupils,masks,support=audit.prepare(shot,side,np.array(report['focal']))
        assert {str(k):v for k,v in support.items()}==r['support']
        with np.load(out/f'designs_{side}.npz') as d:
            for j in range(6):
                assert np.array_equal(matrix[j],d[f'design_{j}'])
                assert np.array_equal(xy[j],d[f'xy_{j}'])
                assert np.array_equal(directions[j],d[f'directions_{j}'])
        saved_read=audit.read_rgb; saved_load=np.load; reads=[]
        def guarded(s,which,frame):
            if frame==115: reads.append(frame); raise AssertionError('Diagnostic RGB leaked into fit')
            return saved_read(s,which,frame)
        class Replaced:
            def __init__(self,obj,value): self.obj,self.value=obj,value
            def __enter__(self): return self
            def __exit__(self,*args): self.obj.close()
            def __getitem__(self,key):
                if key=='linear_rgb': reads.append(key); return np.full_like(self.obj[key],self.value)
                return self.obj[key]
        for value in (0.,1000.):
            def substituted(path,*args,**kwargs):
                obj=saved_load(path,*args,**kwargs)
                return Replaced(obj,value) if f'000115_{side}.npz' in str(path) else obj
            with patch.object(audit,'read_rgb',guarded),patch.object(np,'load',substituted):
                selection,pred=audit.fit_predict(shot,side,obs,matrix,xy,masks)
            assert selection==r['selection']
            with saved_load(out/f'predictions_{side}.npz') as d:
                for key,item in pred.items(): assert np.array_equal(item,d[key])
                assert np.array_equal(d['supported'],masks[5])
                assert np.array_equal(d['labelled_reflection'],reflections[5])
                assert np.array_equal(d['pupil_fraction'],pupils[5])
        assert not reads
        assert audit.score(shot,side,out)==r['seen_815_scores']
        old_root=shot.composite/'r27_photometry'
        with saved_load(old_root/f'designs_{side}.npz') as d: old_data={k:d[k] for k in d.files}
        training=[]
        for j,frame in enumerate((30,60,90,120)):
            a=saved_read(shot,side,frame); p=old_data[f'xy_{j}']; training.append(a[p[:,1],p[:,0]])
        with saved_load(out/f'old_support_{side}.npz') as d:
            for i,record in enumerate(r['old_support_stability']):
                g=record['geometry_index']; x=np.concatenate([old_data[f'design_{j}'][g] for j in range(4)])
                co=audit.fit_radiance(x,np.concatenate(training),record['reflection_ridge'],289)
                assert np.array_equal(co,d['coefficients'][i])
                assert np.array_equal(old_data['design_5'][g]@co,d['predictions'][i])
        transactions=0
        for o in obs:
            source=saved_read(shot,side,o['frame']); mask=np.ones(source.shape[:2],bool)
            output,info=apply_coupled(source,mask,object(),object(),geometry_reliable=False)
            assert np.array_equal(output,source) and info['changed_channels']==0
            transactions+=1
        # Actual surface/layer error diagnostics on the seen source, without refit.
        diagnostics={}
        with saved_load(out/f'predictions_{side}.npz') as d: coefficients=d['coefficients']
        for choice in ('selected','control','best_reflected'):
            i=r['selection'][choice]; g=r['selection']['records'][i]['geometry_index']; co=coefficients[i]
            frames=[]
            for j,o in enumerate(obs):
                source=saved_read(shot,side,o['frame']); p=xy[j]; truth=source[p[:,1],p[:,0]]
                prediction=matrix[j][g]@co
                use=np.ones(len(p),bool) if j<4 else masks[j]
                frames.append(dict(frame=o['frame'],all=audit.error(prediction[use],truth[use]),
                    labelled_reflection=audit.error(prediction[use&reflections[j]],truth[use&reflections[j]])))
            diagnostics[choice]=dict(index=i,backing_rgb=co[289].tolist(),frames=frames)
        results[side]=dict(replayed_models=len(r['selection']['records']),target_rgb_reads=0,
            replayed_old_support_regularized_models=len(r['old_support_stability']),
            target_rgb_substitutions=2,disabled_complete_transactions=transactions,layer_diagnostics=diagnostics)
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--skip-videos',action='store_true'); args=ap.parse_args(); shot=from_args(args)
    from verify_r15 import fallback
    # R27 replay is read-only and verifies that the optional pupil offset leaves
    # every saved old design, fit, score, selection and default transport intact.
    from verify_r27_photometry import replay as old_replay, controls as old_controls
    result=dict(fixture=controls(),source=replay(shot),exact_source=fallback(shot),
                r27_controls=old_controls(),r27_replay=old_replay(shot))
    if not args.skip_videos:
        from verify_r20_eyes import video_checks
        result['videos']={name:video_checks(shot,name,count,fps,shape) for name,count,fps,shape in
            [('r28_radiance',150,float(shot.frame_rate),(960,768,3)),('r28_layers',72,24.,(636,960,3))]}
        import cv2
        for name,frames in [('r28_radiance',(30,60,90,105,115,120)),('r28_layers',(0,24,48))]:
            cap=cv2.VideoCapture(str(shot.final/f'{name}.mp4'))
            for f in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,im=cap.read(); assert ok
                assert cv2.imwrite(str(shot.composite/f'{name}/decoded_{f:03d}.png'),im)
            cap.release()
    result['verifier_sha256']=sha(ROOT/'scripts/verify_r28_radiance.py')
    write(shot.logs/'r28_verification.json',result)
    print('R28_VERIFIED',json.dumps(result['exact_source']),flush=True)


if __name__=='__main__': main()
