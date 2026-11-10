# Integration

Use `ngsr` to call the model engines as separate processes and run the stages in
`scripts/`. Each engine keeps its own Python environment.

## Inputs and outputs

| Stage | Input | Output |
|---|---|---|
| CorridorKey 2.0 | Decoded, desqueezed sRGB/Rec.709 PNGs; optional matte hints | Straight linear Rec.709 foreground and coverage alpha, FLOAT EXR |
| EyeClean | `key/fg/%06d.exr` and exact fps | Signed RGB correction EXRs, index/report and source hashes in `ngsr.json` |
| Normals | Foreground, alpha and validated axis mapping | Unit world normals in `relight/normals/%06d.npz` |
| Proxy | Temporal depth and normals | Mesh caches in `relight/proxy/` |
| Blender | Plate camera, proxy, lights and eye geometry | Irradiance and reflection EXRs |
| Compositor | Those passes, source lighting and optional eye corrections | Review PNGs; `--linear-output` adds premultiplied RGBA FLOAT EXRs |

Frames use zero-based, six-digit filenames within a shot. `selection.start_frame` records
the offset into the original video. Keep fps as a ratio in metadata. CorridorKey's CLI
takes decimal fps; the adapter also saves the original ratio.

For external Blender passes, use shot-local frame numbers or pass `--render-manifest`
with a mapping:

```json
{"per_frame": [{"out_frame": 0, "shot_frame": 0}]}
```

Render with the plate's camera and resolution, linear Rec.709, coverage alpha, and a neutral
diffuse material with specular disabled. The compositor divides out that material's
reflectance: 0.6 by default, or `--render-albedo`. Leave the view transform out of the EXR.
The result is premultiplied RGBA and can be placed over a background with an over operation.

## Python API

`run_keyer(input_frames, output, repository=..., python=..., fps=..., ffmpeg=...)`
calls CorridorKey 2.0 and returns the raw alpha and foreground directories.
`scripts/key_plate.py --shot ...` also preserves source pixels in the opaque core and
applies the shot's despill settings. It refuses to overwrite existing key EXRs.

`run_eyeclean(foreground, output, fps=..., engine=[...])` accepts `on_event`, `cancel`,
`model` and `max_frames`. Set `EYECLEAN_CMD` or `EYECLEAN_PYTHON`, put `eyeclean`
on PATH, or pass the command as an argv list. The adapter writes `ngsr.eyes/1` metadata
after the run succeeds and source hashes are verified. Use a fresh output directory when
rerunning a failed or cancelled job.

`EyeCleanPatches(directory).apply(rgb, frame, source_file)` checks colour, dimensions and
source SHA-256, then adds the float patches on CUDA. It returns corrected float32 RGB,
repaired eyes in subject-view names, and the changed-pixel count. Unknown versions and
missing frames raise errors.

`run_stage(stage, shot, repository=..., python=..., blender=..., options=[...])`
calls a script with an argv list and raises on failure. Choose the interpreter for that stage.

## Shot paths

Shots can live outside the repository. Each needs `shot.json`, `source/frames/` and the
caches required by its stages. For external shots, relative `environment.dir` paths resolve
against the shot directory. Existing shots inside the repository use repository-relative
environment paths. Use absolute paths for shared scene libraries.

Start from `examples/shot.json` and set the screen thresholds, camera, colour and timing
for the plate. Decode V-Log/V-Gamut and desqueeze anamorphic footage upstream.

## Accepted skin and wet-eye settings

This command uses the reporter's accepted settings with matching bulb/fill passes and
fitted eye tracks:

```powershell
python scripts/composite_ratio.py `
  --shot path/to/shot `
  --render-dir path/to/full_irradiance `
  --light-passes "bulb=path/to/bulb,fill=path/to/fill" `
  --name relit --eye-patches path/to/eyeclean `
  --corneal-dir path/to/eye_pass --corneal-blur 0.35 --corneal-matte lids `
  --limbus-track limbus_track_tv_import.json `
  --eyelid-track eyelid_track_wtv40_import.json `
  --meso-sigma 0 --meso-strength 0.5 `
  --specular-model physical --specular 0.5 --roughness-skin 0.55 `
  --f0-skin 0.028 --specular-texture 2 --specular-clamp 8 --linear-output
```

Generate tracks for each new shot; the filenames above belong to the reporter example.
`--meso-sigma 0` measures the proxy/normal crossover. Meso strength controls relief;
specular gain controls shine. The texture factor approximates pore detail. Hair removal stays off.

EyeClean targets iris reflections. To remove studio glints from the white of the eye, run:

```powershell
python scripts/sclera_glint_removal.py --shot path/to/shot `
  --eye-patches path/to/eyeclean --name sclera
```

Then pass `--sclera-pass path/to/shot/composite/sclera` to the compositor.
This pass saves sclera-only float corrections, preserves EyeClean's edited pixels and skips
untouched eyes. It checks the EyeClean run's hash. Tear-lake glints may remain.
