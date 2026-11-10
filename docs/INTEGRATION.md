# Integration

The installable `ngsr` package is a small, file-based integration layer. It does not import
CorridorKey or EyeClean engines into your interpreter. The maintained source stages remain
under `scripts/`; their research predecessors remain available for reference.

## Contracts

| Stage | Input | Output |
|---|---|---|
| CorridorKey 2.0 adapter | decoded, desqueezed sRGB/Rec.709 PNG frames and optional matte hints | straight linear Rec.709 foreground and linear coverage alpha, FLOAT EXR |
| EyeClean adapter | `key/fg/%06d.exr`, exact fps ratio | signed RGB correction EXRs, index/report, source binding `ngsr.json` |
| Normals | keyed foreground/alpha, validated model convention | unit world normals in `relight/normals/%06d.npz` |
| Proxy | temporal depth and normals | mesh caches in `relight/proxy/` |
| Blender | matched plate camera, proxy, scene lights and eye geometry | irradiance/reflection EXRs with frame mapping |
| Ratio compositor | those passes, source light estimate and optional EyeClean run | review PNG; `--linear-output` adds premultiplied RGBA FLOAT EXR |

All frame filenames are local, zero-based six-digit indices. `selection.start_frame` records
their position in the original video. Keep sequence timing exact in metadata; the keyer's
CLI takes decimal fps, but the adapter records the original rational value.

External Blender passes need no bulb-tour log. Use shot-local numbered EXRs, or supply
--render-manifest with {"per_frame":[{"out_frame":0,"shot_frame":0}]}.
The render must use the plate camera/resolution, scene-linear Rec.709, a neutral diffuse
material with specular disabled, and coverage alpha. The compositor divides out neutral
reflectance: default 0.6, configurable with --render-albedo. Do not bake a view transform.
The linear result is premultiplied RGBA; place it over a background with standard over.

## Calling from Python

`run_keyer(input_frames, output, repository=..., python=..., fps=..., ffmpeg=...)`
runs the official CorridorKey 2.0 engine and returns its raw alpha/foreground directories.
`scripts/key_plate.py --shot ...` adds photographic-core preservation and the shot's despill.
It refuses to overwrite existing key EXRs. Create a fresh shot for A/B comparisons.

`run_eyeclean(foreground, output, fps=..., engine=[...])` runs a fresh EyeClean job.
It supports `on_event`, `cancel`, `model` and `max_frames`. Engine discovery uses
EyeClean's `EYECLEAN_CMD`, `EYECLEAN_PYTHON` or PATH conventions.
It writes the versioned `ngsr.eyes/1` binding only after successful processing and source
hash verification. Cancelled/failed runs must be rerun into a fresh directory.

`EyeCleanPatches(directory).apply(rgb, frame, source_file)` is the CUDA patch compositor.
It validates colour, dimensions and source SHA-256 and returns corrected float32 RGB,
the repaired eyes in subject-view names, and a changed-pixel count.
No 8-bit round trip is involved. Unknown output versions or missing frames are errors.

`run_stage(stage, shot, repository=..., python=..., blender=..., options=[...])`
launches an existing stage with shell-free argv and propagates failures.
Use the appropriate interpreter for each stage.

## Shot locations

A shot may live outside this repository. It requires `shot.json`, `source/frames/`
and stage-specific caches. Relative `environment.dir` values for an external shot resolve
against the shot directory; use absolute paths for shared scene libraries.
Existing in-repository shots retain repository-relative environment resolution.

`examples/shot.json` is a minimal configuration. It is a starting point, not a calibration
for arbitrary footage. Screen thresholds, lens/distance assumptions, colour and timing must
describe the actual shot. V-Log / V-Gamut and anamorphic footage need upstream decoding and
desqueezing; the existing ingest is not a universal camera-log importer.

## Accepted skin / wet-eye development preset

For a tour render with matching per-source passes and fitted eye tracks:

```powershell
python scripts/composite_ratio.py --shot path/to/shot --render-dir path/to/full_irradiance --light-passes "bulb=path/to/bulb,fill=path/to/fill" --name relit --eye-patches path/to/eyeclean --corneal-dir path/to/eye_pass --limbus-track limbus_track_tv_import.json --eyelid-track eyelid_track_wtv40_import.json --corneal-matte lids --corneal-blur 0.35 --meso-sigma 0 --meso-strength 0.5 --specular-model physical --specular 0.5 --roughness-skin 0.55 --f0-skin 0.028 --specular-texture 2 --specular-clamp 8 --linear-output
```

Track filenames above refer to the development shot. New shots must generate their own
tracks. `--meso-sigma 0` measures the proxy/normal crossover rather than fixing a pixel scale.
The scene reflection is added only to successfully repaired eyes. EyeClean currently targets
iris reflections; tear-lake glints may remain. Preserve the accepted sclera cleanup with:
python scripts/sclera_glint_removal.py --shot path/to/shot --eye-patches path/to/eyeclean --name sclera
Then add --sclera-pass path/to/shot/composite/sclera to the compositor.
This reuses the accepted detector/fill, exports sclera-only linear FLOAT corrections, and
leaves the EyeClean iris at float precision. It also leaves EyeClean-abstained eyes alone.
The pass is bound to the EyeClean run; stale combinations fail.

Meso strength and skin specular gain are separate controls. Smooth neural normals do not
resolve pores; photographic texture modulation is a heuristic. Hair removal remains disabled.
