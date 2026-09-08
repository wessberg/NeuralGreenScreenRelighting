# Round 4 — in-progress handoff to Claude: dotted edges and tracked catchlights

**Date:** 2026-09-08. **Status: unfinished, explicitly handed off at the artist's request.**
Read round3.md for the previous completed round. Nothing here claims photorealism achieved.

## 1. Artist request and delegation

Track eyes for catchlights. Investigate dotted aliasing on image-left shoulder/hair in
`composite/ratio/000079.png`, and on hand held to ear in `000051.png`. Face in 51 reads
uncanny/smooth; artist wonders about pores or SSS. Otherwise they find it increasingly believable.
They requested Claude **Opus 5, max effort**, then explicitly asked to hand EVERYTHING remaining
to Claude now. Do not broaden this round into unrelated ingest/environment tasks.

## 2. Work done, not committed

- Read round3, AGENTS.md, capture_format.json and current scripts. Working tree was clean initially.
- Inspected original 51 and 79. Real plate texture remains, existing photographed catchlights
  remain; new physically driven catchlights have not been implemented.
- `stress_proxy.py`: added `--only-output 0,51,79` preserving full tour timing;
  `--diagnostic-no-walls`; `--proxy-dir` for sparse alternate meshes (timeline is still taken
  from production proxy dir). Per-output `render_manifest.json` avoids overwriting production
  log on selected-frame runs. Review these modifications for robustness before shipping.
- Rendered 128-sample no-wall control at `composite/r4_no_walls/`. Dots remain.
  Removing walls is NOT the solution and opens SSS/shadow volume.
- `scripts/diagnose_ratio_edges.py` writes controls to `composite/r4_diagnostics/`.
  Four shoulder panels: keyed plate, existing composite, irradiance, no-wall irradiance.
  The grid-pitch corrugated ridge is clear in irradiance and absent in plate.
- Added experimental `scripts/ratio_edges.py`: unpremultiply coverage and nearest-extend from
  eroded interior at full precision. `composite_ratio.py` now has `--edge-radius`,
  `--legacy-edges`, `--only-output`, and per-output `composite_manifest.json`.
  **This is NOT accepted as the final fix. Default currently enables it: revise before delivery.**
  Radius 8 leaves dots. Radius 24 reduces dots but changes legitimate rim lighting/colour.
  Prefer mesh fix below, then decide whether minimal coverage handling is appropriate.
- Experimental composites, preserve originals for comparison:
  `r4_edge_fixed` (assumed source, radius 8); `r4_edge24` (assumed, 24);
  `r4_measured_no_spec` (measured source, no additive specular, radius 8).
  All contain selected 0/51/79 only. No full new sequence/video generated.

## 3. Strong root cause and mesh fix awaiting visual verification

Independent local reviewer found `make_proxy_mesh.build` adds `Yd=highpass(...)`, whose detail
is multiplied by matte, to a mesh that extends BEYOND matte. Thus detail abruptly drops to zero
at matte boundary, creating a corrugated depth ridge at the photographed silhouette.

Reviewer modified ONLY `scripts/make_proxy_mesh.py`: nearest-extend normal detail from valid
matte into padded geometry before adding it. Added `--output-dir` and `--only-frames` controls.
Generated alternate meshes for 0,51,79 under
`relight/proxy_r4_detail_extension/`; production meshes/log were not changed.

Measured against existing production meshes:

| frame | median adjacent depth jump across matte, before→after | p95 before→after |
|---|---|---|
| 0 | 34.0→6.0 mm | 49.0→18.1 mm |
| 51 | 34.9→7.0 mm | 54.0→21.3 mm |
| 79 | 33.6→6.8 mm | 52.2→22.1 mm |

Every front vertex INSIDE matte unchanged bit-for-bit (max XYZ delta 0.0). Padding changes
reach 60–79 mm. Residual 6–7mm jump may warrant extending final Yw instead; test, do not assume.
These are reviewer-reported measurements; ensure persistent report exists before final handoff.

Rendered these fixed meshes successfully at 128 samples, closed walls, selected output
0,51,79, into `composite/r4_mesh_fixed/`. **Not yet viewed or composited** at handoff.
Blender process completed; console log `logs/r4_mesh_fixed.txt`.

Command used:
```
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' -b --factory-startup --python scripts/stress_proxy.py -- --shot music_videos/reporter_test/shots/reporter_a --ratio --length 150 --only-output 0,51,79 --samples 128 --proxy-dir music_videos/reporter_test/shots/reporter_a/relight/proxy_r4_detail_extension --name r4_mesh_fixed
```

## 4. Eyes — entirely outstanding

`face_track.py` already estimates 478 landmarks but discards iris centres 468/473 after IPD.
Store centres, iris contours, eyelid polygons, blink/visibility gates, coordinate metadata and
missing detection state. Preserve existing scale consumers. Run isolated facetrack venv.
Use tracked small corneal geometry for true reflections of actual lights, separately additive
in composite; DO NOT contaminate irradiance ratio with eye glossy or replace photograph.
Blink/eyelid and proxy occlusion, resolution mapping, source-frame/tour mapping need validation.
Round3 gives proposed approach; use judgment and measured controls, no painted fixed dots.

## 5. Facial smoothness — diagnosis outstanding

Do compare matching plate, ratio-only, ratio+specular, assumed versus measured source lighting
on 51 and other light directions. Added smooth specular can dilute photographed texture.
No evidence yet that absent pores or absent SSS is root cause: Cycles SSS is already folded
into diffuse ratio; coarse shell affects shadow shape. Do not invent skin detail or claim
ratio rendering recovers correct skin transport at all angles.

## 6. Operational handoff

CLI `claude.exe` v2.1.263 exists. First sandboxed `claude -p --model claude-opus-5 --effort max`
returned API ConnectionRefused after ~169s with zero tokens. Escalated retry for eye-only task
returned no output for several minutes and was interrupted to supersede it with full handoff.
No Claude eye edits appeared. Full handoff launch is being attempted by Codex separately.
Do not say Claude performed work unless the launch actually succeeds.

No source footage or key thresholds changed. No commits, full verification, full sequence,
new video, or catchlights yet. Experimental edits must be reviewed, not blindly accepted.
Use proper permissions, no permission bypass, no push/publish or external messaging.

## 7. Starter prompt for continuation / round 5

> Continue unfinished round4 from this handoff. First inspect r4_mesh_fixed render controls
> and composite them with legacy edge handling to isolate the geometry fix. Confirm dotted
> shoulder/hand improvement and temporal stability, preserve original controls. Complete
> tracked, occluded, light-driven catchlights and diagnose smooth face using controlled
> comparisons. Regenerate the 150-frame review sequence only after corrections validated.
> Review all changes, write evidence and next round5 handoff including honest limitations,
> then commit with why. Do not report full photorealism while constraints remain unsatisfied.
