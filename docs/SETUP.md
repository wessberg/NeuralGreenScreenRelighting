# Setup

Tested on Windows with an NVIDIA RTX 5080. The adapters support Linux paths; the full
pipeline has not been tested there. Run GPU stages one at a time.

## Fetch the source dependencies

`dependencies.lock.json` pins the engine revisions and model files.

```
python scripts/fetch_dependencies.py
```

The script clones into `dependencies/`, checks out the pinned commits and refuses to
change tracked files with local edits. Weights are downloaded separately.

## Pipeline environment

Use Python 3.13 and CUDA 12.8:

```powershell
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe -e ".[pipeline,dev]" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The old `dependencies/CorridorKey/.venv` still runs the geometry and compositor scripts.
Keying runs in `dependencies/CorridorKey2/.venv`. Keep its requirements in that environment.

Sapiens2 downloads the pinned `facebook/sapiens2-normal-1b` checkpoint on first inference.
Validate the normal axes before using a new checkpoint or camera convention:

```powershell
.venv/Scripts/python.exe scripts/validate_normals_convention.py --shot path/to/shot
.venv/Scripts/python.exe scripts/neural_normals.py --shot path/to/shot --convention logs/normals_convention.json
```

`configs/normals-convention.json` records the mapping for the existing checkpoint.

Face tracking needs a separate MediaPipe environment; the development checkout uses
`dependencies/facetrack`. Depth needs [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
at `dependencies/VideoDepthAnything` with its Small checkpoint. Follow that project's
setup instructions. The adapter package doesn't install these stages.

## CorridorKey 2.0

CorridorKey's NumPy and Transformers requirements conflict with the normals pipeline.
Install it with Python 3.12:

```powershell
uv venv dependencies/CorridorKey2/.venv --python 3.12 --seed
dependencies/CorridorKey2/.venv/Scripts/python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
dependencies/CorridorKey2/.venv/Scripts/python.exe -m pip install -r dependencies/CorridorKey2/requirements.txt
dependencies/CorridorKey2/.venv/Scripts/python.exe scripts/download_models.py
```

The download is roughly 5 GB, including C-RADIO, MoGe and RVM. Put FFmpeg on PATH or pass
`--ffmpeg /path/to/ffmpeg` to `scripts/key_plate.py`.

The wrapper reads native-resolution, linear FLOAT EXRs from the engine. `--test` processes
the first eight consecutive frames so the temporal model has context.

## EyeClean

```powershell
cd dependencies/eyeclean
./bootstrap.ps1
uv run eyeclean models
cd ../..
$env:EYECLEAN_PYTHON = "$PWD/dependencies/eyeclean/.venv/Scripts/python.exe"
ngsr eyes --shot path/to/shot --output path/to/fresh/eyeclean
```

EyeClean needs Python 3.12, CUDA 13 and a compatible driver. On Linux, use
`bootstrap.sh` and `.venv/bin/python`.

The pinned registry has no download URL for r57. Supply `eyeclean-r57.safetensors` in
`EYECLEAN_MODEL_DIR` with this SHA-256:

```
e1ab7abbbe409d7a9f1815148fc00b337999e04c6c1de2bacf191c67cd2799bb
```

`models --fetch` cannot download it from that registry. The reporter example uses an
installed copy. Public model distribution is still on the [release checklist](RELEASE.md).

The package pins the client separately from the engine. Each run records the model name and hash.

## Blender

Use Blender 5.2.1 with factory settings, Cycles/OptiX and persistent render data disabled:

```powershell
& "/path/to/blender" -b --factory-startup --python scripts/stress_proxy.py -- --shot path/to/shot --ratio
& "/path/to/blender" -b --factory-startup --python scripts/stress_proxy.py -- --shot path/to/shot --corneal --globe --socket --name eye_pass
```

These renders need depth, proxy geometry, face tracks and fitted eye tracks.
The bulb tour is an example rig. Other scenes must supply aligned irradiance and reflection
passes; the current per-light skin settings use the tour's bulb/fill metadata.
