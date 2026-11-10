# Reporter development example

Run from a configured checkout. The source reporter shot and accepted Blender caches must
already be present; they are local development fixtures, not shipped media.

```powershell
$env:EYECLEAN_PYTHON = "$PWD/dependencies/eyeclean/.venv/Scripts/python.exe"
dependencies/CorridorKey/.venv/Scripts/python.exe examples/reporter_smoke.py --ffmpeg dependencies/ffmpeg/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe
```

The script creates `runs/reporter-release/`, keys the selected contiguous excerpt with
CorridorKey 2.0, runs EyeClean r57, regenerates normals, reuses the already accepted
same-source proxy and Blender passes, and runs the skin/wet-eye compositor.
It never writes the accepted shot's key, normals or composites.

Output includes `review/` stills, tight eye comparisons, a contact sheet, a short
source/accepted/new comparison video, linear EXRs, and a JSON report.
A separate eye comparison video uses the exact source frame rate for temporal inspection.
A cached-pass smoke run proves the updated stages can be combined; it does not validate a
fresh Blender scene build, every frame of the original 20-second clip, or artistic acceptance.
Use `--frames` to shorten the excerpt. View consecutive frames to assess temporal changes.

For keying comparisons without relighting, run `python examples/review_reporter_key.py`.
It compares the old key, raw v2 and adapted v2 on white, grey and black, with edge crops.
The reporter results look very similar to v1. A matched eight-frame native no-hint control
also did not establish a visible quality gain. Changed pixel counts are not a quality verdict.
