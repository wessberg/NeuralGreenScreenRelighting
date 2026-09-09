"""Download pinned author SVD DiffusionRenderer weights for a selective --shot trial."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from shot_paths import ROOT, add_shot_argument, from_args


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap); args=ap.parse_args(); shot=from_args(args)
    os.environ['HF_HOME']=str(ROOT/'dependencies/diffusion-runtime/hf-cache')
    os.environ['HF_HUB_DISABLE_XET']='1'
    from huggingface_hub import HfApi, snapshot_download
    destination=ROOT/'dependencies/diffusion-weights'; destination.mkdir(exist_ok=True)
    result={}
    for kind in ('inverse','forward'):
        repo=f'nexuslrf/diffusion_renderer-{kind}-svd'; folder=destination/kind
        pin=destination/f'{kind}_pin.json'
        revision=json.loads(pin.read_text())['revision'] if pin.exists() else None
        info=HfApi(token=False).model_info(repo,revision=revision,files_metadata=True)
        files=[s for s in info.siblings if s.rfilename.endswith(('.json','.safetensors','.bin','.md','.txt'))]
        # Each component needs one weight format. Prefer safetensors when present.
        safe_dirs={str(Path(s.rfilename).parent) for s in files if s.rfilename.endswith('.safetensors')}
        files=[s for s in files if not(s.rfilename.endswith('.bin') and str(Path(s.rfilename).parent) in safe_dirs)]
        record=dict(repository=repo,revision=info.sha,files=[dict(path=s.rfilename,bytes=s.size,
            expected_sha256=s.lfs.sha256 if s.lfs else None) for s in files])
        pin.write_text(json.dumps(record,indent=2)+'\n')
        print('DOWNLOAD',kind,info.sha,sum(s.size or 0 for s in files)/2**30,'GiB',flush=True)
        snapshot_download(repo,revision=info.sha,local_dir=folder,allow_patterns=[s.rfilename for s in files],
                          token=False,max_workers=3)
        for row in record['files']:
            actual=sha(folder/row['path']); row['sha256']=actual
            if row['expected_sha256']: assert actual==row['expected_sha256'],row['path']
        result[kind]=record
        print('VERIFIED_WEIGHTS',kind,flush=True)
    (shot.logs/'r31_weights.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
