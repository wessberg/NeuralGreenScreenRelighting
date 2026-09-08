"""Reporter-only frozen RAFT correspondence/blink diagnostic; no production repair."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from audit_r14_neural import load_rows, SIDES, tile, write
from dense_iris_flow import DenseEyeFlow, assess_pair, aggregate, query_flow
from ellseg_iris import EllSeg
from ritnet_iris import RITnet
from neural_iris_correspondence import observe
from shot_paths import add_shot_argument, from_args
from temporal_iris import apply_eye_repair, footprint_valid

ROOT = Path(__file__).resolve().parents[1]
OFFSETS = (-12, -8, -4, 4, 8, 12)


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return plain(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def digest(obj):
    return hashlib.sha256(json.dumps(plain(obj), sort_keys=True).encode()).hexdigest()


def stats(values):
    return dict(count=len(values), median=float(np.median(values)), p95=float(np.percentile(values,95))) if values else None


def evaluate_case(row, donors, points, flow, em, rm):
    query = np.array(points, np.float32)
    hidden = np.zeros(row['rgb'].shape[:2], bool)
    hidden[query[:,1].astype(int), query[:,0].astype(int)] = True
    target = observe(row['rgb'], em, rm, excluded=hidden, track_found=row['track_found'])
    scrubbed = row['rgb'].copy()
    scrubbed[hidden] = 0
    pairs, fingerprint, fields = [], [], {}
    for donor in donors:
        fwd, back = flow.pair(row['rgb'], donor['rgb'], target_excluded=hidden)
        pair = assess_pair(scrubbed, donor['rgb'], target, donor['ob'], query, fwd, back)
        pair['donor_frame'] = donor['frame']
        pairs.append(pair)
        fingerprint.extend([hashlib.sha256(a.tobytes()).hexdigest() for a in (fwd,back)])
        fields[f"{donor['frame']}_forward"] = fwd
        fields[f"{donor['frame']}_backward"] = back
    output = dict(target=dict(candidate_ok=target.candidate_ok, reasons=target.reasons,
        metrics=target.metrics), pairs=pairs,
        aggregates={mode:{method:aggregate(pairs,method,mode) for method in ('fixed','raft')}
                    for mode in ('inbound','candidate','gated')})
    fingerprint.append(digest(asdict(target)))
    fingerprint.append(digest(output))
    return output, digest(fingerprint), fields


def summarize(cases):
    result = {}
    for mode in ('inbound','candidate','gated'):
        result[mode] = {}
        paired_errors = {'fixed':[], 'raft':[]}
        for method in ('fixed','raft'):
            errors, supported, agreement = [], 0, 0
            for case in cases:
                a = case['result']['aggregates'][mode][method]
                truth = np.asarray(case['truth'])
                valid = np.asarray(a['supported'])
                paired = valid & np.asarray(case['result']['aggregates'][mode]['raft' if method=='fixed' else 'fixed']['supported'])
                err = np.abs(np.asarray(a['rgb'])-truth).mean(-1)
                errors.extend(err[valid].tolist())
                paired_errors[method].extend(err[paired].tolist())
                supported += int(valid.sum())
                agreement += int(np.asarray(a['agreement']).sum())
            result[mode][method] = dict(supported_pixels=supported, agreement_pixels=agreement,
                per_pixel_display_mae=stats(errors), paired_per_pixel_display_mae=stats(paired_errors[method]))
    return result


def render_review(shot, rows, blink):
    lookup = {(r['frame'],r['side']):r for r in blink}
    for k,i in enumerate(range(950,1000)):
        panels=[]
        for side in SIDES:
            rgb=rows[side][i]['rgb']
            rec=lookup[i,side]
            overlay=rgb.copy()
            for xy,cyc,mat in zip(rec['mapped'],rec['roundtrip'],rec['candidate']):
                if np.isfinite(xy).all():
                    color=(0,255,0) if cyc is not None and cyc<=.5 and mat else (255,70,30)
                    cv2.circle(overlay,tuple(np.rint(xy).astype(int)),1,color,-1)
            result,info=apply_eye_repair(rgb,np.ones(rgb.shape[:2],bool),geometry_reliable=False)
            assert info['changed_channels']==0 and np.array_equal(rgb,result)
            n=sum(v is not None and v<=.5 for v in rec['roundtrip'])
            panels.append(np.concatenate([
                tile(rgb,f'{side} SOURCE {i}','original room reflection'),
                tile(overlay,'RAFT POINTS FROM 977',f'cycle <=0.5px: {n}/5; NOT visibility'),
                tile(result,'EXACT SOURCE FALLBACK','NO REMOVAL / REPLACEMENT')],1))
        write(shot.composite/'r15_dense/review'/f'{k:06d}.png',np.concatenate(panels,0))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--review-only',action='store_true')
    args=ap.parse_args()
    shot=from_args(args)
    # Evaluation loader relies on reporter-specific source cache coordinates.
    if shot.dir.resolve() != (ROOT/'music_videos/reporter_test/shots/reporter_a').resolve():
        raise SystemExit('Reporter-only audit; use dense_iris_flow array API for other shots')
    labels=json.loads((ROOT/'scripts/data/r14_native_labels.json').read_text())
    ids=set(range(950,1000))
    ids.update(l['frame']+d for l in labels['frames'] for d in OFFSETS)
    rows=load_rows(shot,sorted(ids))
    report_path=shot.logs/'r15_dense.json'
    if args.review_only:
        render_review(shot,rows,json.loads(report_path.read_text())['blink'])
        return
    flow,em,rm=DenseEyeFlow(),EllSeg(),RITnet()
    out=shot.composite/'r15_dense'
    out.mkdir(parents=True,exist_ok=True)
    for side in SIDES:
        for row in rows[side].values():
            row['ob']=observe(row['rgb'],em,rm,track_found=row['track_found'])
        print('proposals',side,flush=True)
    cases,probes=[],[]
    for label in labels['frames']:
        i,side=label['frame'],label['side']
        row=rows[side][i]
        donors=[rows[side][i+d] for d in OFFSETS]
        for category in ('iris','reflection'):
            points=label[category]
            result,reference,fields=evaluate_case(row,donors,points,flow,em,rm)
            invariance=[]
            for color in ([0,0,0],[255,0,255]):
                altered=row['rgb'].copy()
                for x,y in points:
                    altered[y,x]=color
                _,fingerprint,_=evaluate_case(dict(row,rgb=altered),donors,points,flow,em,rm)
                assert reference==fingerprint,(i,side,category,color)
                invariance.append(dict(color=color,exact=True))
            np.savez_compressed(out/f'{i}_{side}_{category}_flows.npz',**fields)
            case=dict(frame=i,side=side,points=points,donor_ids=[i+d for d in OFFSETS],
                hidden_rgb_invariance=invariance,result=result)
            if category=='iris':
                case['truth']=[row['rgb'][y,x].tolist() for x,y in points]
                cases.append(case)
            else:
                case['clean_truth_known']=False
                case['pupil_crossing']=label['pupil_crossing']
                probes.append(case)
            print('evaluated',i,side,category,flush=True)
    blink=[]
    for side in SIDES:
        anchor=rows[side][977]
        points=next(l['iris'] for l in labels['frames'] if l['frame']==977 and l['side']==side)
        query=np.array(points,np.float32)
        for i in range(950,1000):
            row=rows[side][i]
            fwd,back=flow.pair(anchor['rgb'],row['rgb'])
            mapped,inbound,cycle=query_flow(fwd,back,query)
            candidate=inbound & footprint_valid(row['ob'].material,mapped)
            blink.append(plain(dict(frame=i,side=side,mapped=mapped,roundtrip=cycle,
                candidate=candidate,candidate_ok=row['ob'].candidate_ok,
                reasons=row['ob'].reasons,metrics=row['ob'].metrics)))
        print('blink sequence',side,flush=True)
    report=plain(dict(protocol_sha256=hashlib.sha256((ROOT/'docs/r15-evaluation-protocol.md').read_bytes()).hexdigest(),
        annotations_sha256=hashlib.sha256((ROOT/'scripts/data/r14_native_labels.json').read_bytes()).hexdigest(),
        model=flow.provenance,attempted_iris_pixels=30,attempted_reflection_pixels=12,
        labels_status='reused seen assistant labels; not independent clean-material truth',
        source_cache='r10 source crops; fixed tracker/crop outside hidden-RGB guarantee',
        context_scope='observed context scoring, not held out from frozen neural receptive field',
        summary=summarize(cases),cases=cases,reflection_probes=probes,blink=blink,
        production_repairs=0,scene_reflection_allowed=False))
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    render_review(shot,rows,blink)
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    main()
