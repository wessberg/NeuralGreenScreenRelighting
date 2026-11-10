# Reporter example

This example needs the local reporter shot and its existing Blender caches.
The repository doesn't include the footage or rendered passes.

```powershell
$env:EYECLEAN_PYTHON = "$PWD/dependencies/eyeclean/.venv/Scripts/python.exe"
dependencies/CorridorKey/.venv/Scripts/python.exe examples/reporter_smoke.py --ffmpeg dependencies/ffmpeg/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe
```

The script writes to `runs/reporter-release/`. It runs CorridorKey 2.0, EyeClean r57 and
Sapiens2 again, then uses the existing proxy, eye tracks and Blender passes to composite
the actor with the accepted skin settings. The original shot stays unchanged.

`review/` contains stills, eye crops, a contact sheet and two videos. The full comparison
runs at 24 fps for the light tour; the eye comparison uses the source frame rate.
Linear EXRs and a JSON report sit alongside the stage outputs.

Use `--frames` for a shorter excerpt or `--resume` to reuse completed stages.
The run covers 150 frames from the 20-second clip and reuses rendered passes. A fresh scene
build and the full clip still need testing.

For unrelit keying comparisons:

```
python examples/review_reporter_key.py
```

This writes old-key, raw-v2 and adapted-v2 comparisons on white, grey and black, plus edge
crops. The v1 and v2 results look very similar on this plate. An eight-frame v2 run without
hints also showed no clear gain.
