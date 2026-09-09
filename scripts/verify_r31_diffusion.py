"""Verify the actual selective trial, preserved source, conditioning and encoded review."""
import argparse
import ast
import json
from pathlib import Path
from diffusion_eye_trial import ROOT, PASSES, sha, write, rgb8
import cv2
import numpy as np
from shot_paths import add_shot_argument, from_args


def verify(shot, name, videos):
    root=shot.composite/name
    meta=json.loads((root/'input.json').read_text())
    lo=np.array(meta['crop_xyxy'][:2]); hi=np.array(meta['crop_xyxy'][2:])
    with np.load(root/'source.npz') as z: source,masks=z['linear'],z['masks']
    assert source.shape[:3]==masks.shape and np.isfinite(source).all()
    assert np.isfinite(masks).all() and masks.min()>=0 and masks.max()<=1
    assert (masks>0).any(), 'A no-op is not an eye-regeneration trial'
    with np.load(root/'selective_composites.npz') as z: composites=dict(z)
    untouched=0; closed=[]; differences={k:[] for k in composites}
    for j,row in enumerate(meta['rows']):
        p=shot.fg(row['frame']); assert sha(p)==row['source_sha256']
        original=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)[...,::-1]
        assert np.array_equal(original[lo[1]:hi[1],lo[0]:hi[0]],source[j])
        if 'model_input_sha256' in row:
            assert sha(root/'input'/f'{row["frame"]:06d}.png')==row['model_input_sha256']
        outside=masks[j]==0
        untouched+=int(outside.sum())
        if outside.all(): closed.append(row['source_frame'])
        for k,values in composites.items():
            assert values.shape==source.shape and np.isfinite(values[j]).all()
            assert np.array_equal(values[j][outside],source[j][outside])
            if (~outside).any():
                differences[k].append(float(np.abs(rgb8(values[j]).astype(float)-rgb8(source[j]))[~outside].mean()))
    assert all(max(d)>0 for d in differences.values())
    records=[]
    for kind,labels in [('inverse',PASSES),('forward',('neutral','scene'))]:
        for label in labels:
            folder=root/kind/label; record=json.loads((folder/'complete.json').read_text())
            assert len(record['image_sha256'])==len(meta['rows'])
            for p,h in record['image_sha256'].items():
                assert sha(folder/p)==h
                im=cv2.imread(str(folder/p)); assert im.shape==(meta['inference_size'],meta['inference_size'],3)
            records.append(record)
    # Execute the released direction functions independently of our NumPy port.
    import torch
    torch.set_num_threads(4)
    author=ROOT/'dependencies/DiffusionRenderer/src/data/rendering_utils.py'
    tree=ast.parse(author.read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('latlong_vec','envmap_vec')]
    namespace=dict(torch=torch,np=np)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(author),'exec'),namespace)
    expected=namespace['envmap_vec']((512,512),device='cpu').numpy()
    with np.load(root/'environments.npz') as z:
        assert all(v.shape==(512,512,3) and np.isfinite(v).all() and v.min()>=0 and v.max()<=1 for v in z.values())
        maximum=float(np.abs(z['scene_env_nrm']*2-1-expected).max())
        assert maximum<1e-6
        assert not np.array_equal(z['neutral_env_ldr'],z['scene_env_ldr'])
    result=dict(frames=len(meta['rows']),native_unchanged_pixels_per_arm=untouched,
        full_closure_exact_source_frames=closed,all_source_hashes_unchanged=True,
        all_generated_image_hashes_match=True,all_arrays_finite=True,
        author_environment_direction_max_abs_error=maximum,
        mean_display_change_inside_masks={k:float(np.mean(v)) for k,v in differences.items()},
        display_change_is_not_a_quality_score=True,
        inference_seconds=sum(r['elapsed_seconds'] for r in records),
        max_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in records))
    if videos:
        path=shot.final/f'{name}.mp4'; cap=cv2.VideoCapture(str(path))
        fps=cap.get(cv2.CAP_PROP_FPS); count=0; errors=[]; eye_rows=[]
        while True:
            ok,frame=cap.read()
            if not ok: break
            original=cv2.imread(str(root/'review'/f'{count:06d}.png'))
            assert original is not None and frame.shape==original.shape==(600,1280,3)
            errors.append(float(np.abs(frame.astype(float)-original.astype(float)).mean()))
            strip=frame[370:550].copy()
            cv2.putText(strip,f'frame {count}',(3,16),0,.45,(0,255,255),1,cv2.LINE_AA)
            eye_rows.append(strip)
            if count in (0,5,8,9,12,17,23):
                assert cv2.imwrite(str(root/f'decoded_{count:03d}.png'),frame)
            count+=1
        cap.release()
        assert count==len(meta['rows']) and abs(fps-float(shot.frame_rate))<.001
        assert max(errors)<3
        for start in range(0,count,8):
            assert cv2.imwrite(str(root/f'decoded_eyes_{start:03d}.png'),np.concatenate(eye_rows[start:start+8]))
        result['video']=dict(frames=count,fps=fps,seconds=count/fps,sha256=sha(path),
                             max_frame_mean_encode_error=max(errors),all_frames_decoded=True)
        slow=shot.final/f'{name}_slow.mp4'
        if slow.exists():
            cap=cv2.VideoCapture(str(slow)); slow_fps=cap.get(cv2.CAP_PROP_FPS); slow_errors=[]
            while True:
                ok,frame=cap.read()
                if not ok: break
                original=cv2.imread(str(root/'review_slow'/f'{len(slow_errors):06d}.png'))
                assert original is not None and frame.shape==original.shape==(600,1280,3)
                slow_errors.append(float(np.abs(frame.astype(float)-original.astype(float)).mean()))
            cap.release()
            assert len(slow_errors)==count and slow_fps==12 and max(slow_errors)<3
            result['slow_video']=dict(frames=count,fps=slow_fps,sha256=sha(slow),
                                      max_frame_mean_encode_error=max(slow_errors),all_frames_decoded=True)
    write(shot.logs/f'{name}_verification.json',result)
    print(json.dumps(result,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--name',default='r31_diffusion512'); ap.add_argument('--videos',action='store_true')
    ap.add_argument('--spots',action='store_true')
    args=ap.parse_args(); shot=from_args(args)
    if Path(args.name).name!=args.name: raise ValueError('Expected single trial directory name')
    if args.spots: verify_spots(shot,args.name)
    else: verify(shot,args.name,args.videos)


def verify_spots(shot,name):
    root=shot.composite/name; record=json.loads((shot.logs/f'{name}.json').read_text())
    parent=shot.composite/record['input_trial']
    with np.load(parent/'source.npz') as z: source=z['linear']
    with np.load(root/'patches.npz') as z: result,masks=z['linear'],z['masks']
    assert result.shape==source.shape and masks.shape==source.shape[:3]
    assert np.isfinite(result).all() and np.isfinite(masks).all() and masks.min()>=0 and masks.max()<=1
    assert np.array_equal(result[masks==0],source[masks==0]) and not np.array_equal(result,source)
    for row in record['rows']:
        assert sha(shot.fg(row['frame']))==row['source_sha256']
        assert sha(parent/'inverse/basecolor'/f'{row["frame"]:06d}.png')==row['generated_basecolor_sha256']
    videos={}
    for suffix,folder,fps in [('', 'review',float(shot.frame_rate)),('_slow','review_slow',12.)]:
        p=shot.final/f'{name}{suffix}.mp4'; cap=cv2.VideoCapture(str(p)); actual_fps=cap.get(cv2.CAP_PROP_FPS)
        frames=[]; errors=[]
        while True:
            ok,frame=cap.read()
            if not ok: break
            original=cv2.imread(str(root/folder/f'{len(frames):06d}.png'))
            assert original is not None and frame.shape==original.shape==(640,1152,3)
            errors.append(float(np.abs(frame.astype(float)-original.astype(float)).mean()))
            strip=frame[426:586].copy()
            cv2.putText(strip,str(len(frames)),(4,16),0,.45,(0,255,255),1,cv2.LINE_AA)
            frames.append(strip)
        cap.release(); assert len(frames)==len(source) and abs(actual_fps-fps)<.001 and max(errors)<3
        if not suffix:
            for start in range(0,len(frames),8):
                assert cv2.imwrite(str(root/f'decoded_eyes_{start:03d}.png'),np.concatenate(frames[start:start+8]))
        videos[suffix or 'native']=dict(frames=len(frames),fps=actual_fps,sha256=sha(p),max_frame_mean_encode_error=max(errors))
    counts=(masks>0).sum((1,2)); edited=masks>0
    summary=dict(frames=len(source),edited_frames=int((counts>0).sum()),fallback_frames=int((counts==0).sum()),
        maximum_edited_native_pixels_per_frame=int(counts.max()),total_edited_native_pixels=int(counts.sum()),
        exact_source_outside_mask=True,source_hashes_unchanged=True,all_arrays_finite=True,
        edited_mean_source_display=float(rgb8(source)[edited].mean()),edited_mean_result_display=float(rgb8(result)[edited].mean()),
        brightness_change_is_not_clean_material_evidence=True,scene_replacement=False,videos=videos)
    write(shot.logs/f'{name}_verification.json',summary); print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
