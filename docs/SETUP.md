# Setup

Windows with an NVIDIA RTX 5080 is the development platform. Linux paths are supported
by the adapters but the complete pipeline has not been validated there.
Use one GPU stage at a time. No paid inference API or API key is required.

## Source and engine pins

`dependencies.lock.json` records the tested source revisions. Fetch those interfaces with:

```
python scripts/fetch_dependencies.py
```

It clones into ignored `dependencies/`, checks out pinned revisions, and refuses to
change an existing checkout with local modifications. No weights are redistributed.

## Pipeline Python

Use Python 3.13 and CUDA 12.8. A fresh environment avoids tying this project to CorridorKey 1:

```powershell
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe -e ".[pipeline,dev]" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The previous `dependencies/CorridorKey/.venv` can still run the geometry/compositor scripts.
Do not sync it against CorridorKey 2.0's requirements. EyeClean is a third environment.

Sapiens2 downloads `facebook/sapiens2-normal-1b` at the pinned revision in dependencies.lock.json on first inference. Its licence is separate.
The normal convention is validated evidence, not a guessed axis flip:

```powershell
.venv/Scripts/python.exe scripts/validate_normals_convention.py --shot path/to/shot
.venv/Scripts/python.exe scripts/neural_normals.py --shot path/to/shot --convention logs/normals_convention.json
```

The bundled convention record in `configs/` covers the existing checkpoint and world axes;
revalidate if the checkpoint or camera convention changes.

Face tracking uses MediaPipe in an isolated environment (`dependencies/facetrack` on the
development machine). Depth uses the separately fetched Video Depth Anything checkout
at `dependencies/VideoDepthAnything` and its Small checkpoint; see that upstream's setup.
These prerequisites are still source-checkout stages, not automatically provisioned by
the lightweight package.

## CorridorKey 2.0

Its NumPy/Transformers requirements conflict with the normals pipeline. Use Python 3.12:

```powershell
uv venv dependencies/CorridorKey2/.venv --python 3.12 --seed
dependencies/CorridorKey2/.venv/Scripts/python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
dependencies/CorridorKey2/.venv/Scripts/python.exe -m pip install -r dependencies/CorridorKey2/requirements.txt
dependencies/CorridorKey2/.venv/Scripts/python.exe scripts/download_models.py
```

The upstream downloader fetches roughly 5 GB, including C-RADIO, MoGe and RVM.
Keep FFmpeg on PATH, or pass `--ffmpeg /path/to/ffmpeg` to `scripts/key_plate.py`.
The wrapper runs native-resolution temporal inference and reads unclamped linear FLOAT EXRs.
`--test` means the first eight contiguous frames; sparse-frame inference would defeat temporal context.

## EyeClean

```powershell
cd dependencies/eyeclean
./bootstrap.ps1
uv run eyeclean models
cd ../..
$env:EYECLEAN_PYTHON = "$PWD/dependencies/eyeclean/.venv/Scripts/python.exe"
ngsr eyes --shot path/to/shot --output path/to/fresh/eyeclean
```

At the pinned EyeClean revision, r57's registry has no download URL. A fresh installation
needs the separately supplied eyeclean-r57.safetensors bundle in EYECLEAN_MODEL_DIR
(SHA-256 e1ab7abbbe409d7a9f1815148fc00b337999e04c6c1de2bacf191c67cd2799bb).
This is a release prerequisite; models --fetch cannot obtain it from that registry.
The local example uses the already installed bundle, without uploading it.

EyeClean requires its Python 3.12 / CUDA 13 environment and compatible driver.
On Linux use its `bootstrap.sh` and `.venv/bin/python`.
The client is pinned with the adapter package; the engine is deliberately not a Python
dependency of the main process. Model names and hashes are recorded in every run.

## Blender

Blender 5.2.1 is the tested version. Run scenes with factory settings, Cycles/OptiX and
persistent render data disabled. No MCP service is required.

```powershell
& "/path/to/blender" -b --factory-startup --python scripts/stress_proxy.py -- --shot path/to/shot --ratio
& "/path/to/blender" -b --factory-startup --python scripts/stress_proxy.py -- --shot path/to/shot --corneal --globe --socket --name eye_pass
```

Depth, proxy geometry, face and fitted eye tracks must exist before these renders.
The bulb tour is an example rig. A general Blender environment integration must supply
aligned irradiance and reflection passes; the current per-source skin preset understands
the tour's bulb/fill metadata.
