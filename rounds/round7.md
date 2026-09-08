# Round 7 handoff — rejected specular removal and catchlights disabled by default

**Date:** 2026-09-08. Supersedes round6's original starter prompt. Read the rejection addendum
in `round6.md` for the preserved incident record, detailed diagnosis, measurements and videos.

## 1. Delivered and verified

The artist rejected round 5 for destroyed teeth, checkerboard sclera, unstable dark-hair
patches and reflections painted over eyelids. `composite_ratio.py` now defaults to keeping
the original plate and disabling catchlight replacement. Keep mode bypasses specular caches.
The experimental path is labelled rejected and remains explicitly opt-in for investigation.

Fresh `composite/ratio_r7` was computed from the stored irradiance renders. All 150 frames
match the previous good state, `composite/ratio`, exactly: **zero changed 8-bit channels**.
No Blender rerender or key-threshold change was needed.

`audit_r5_defects.py` adds original-plate anatomical integrity measurements and motion-
compensated temporal measurement of hair removal. Rejected round 5 removed **26.88% tooth
candidate energy and 52.66% sclera candidate energy**. Its hair removal changes by **6.30
percentage points** per adjacent frame after optical flow; removal is zero in the rollback.
These figures describe a regression and its removal, not successful physical separation.

## 2. Review artefacts

In `music_videos/reporter_test/shots/reporter_a/final/`, all 150 frames at 24 fps, native pixel
scale, round 4 on the left:

- `r7_teeth_1to1.mp4`
- `r7_hair_1to1.mp4`
- `r7_sclera_1to1.mp4`
- `r7_eyelid_1to1.mp4`

Generate with `review_r5_rollback.py --shot <shot>`. Each video has its own JSON sidecar.
All seven old/new videos decoded completely. Ordered frame sheets of the rejected videos
and selected rollback intervals were inspected, including the requested frame numbers.
There was no real-time playback tool; do not misreport this as real-time viewing or artist
approval. `composite/r7_audit/` retains the visual evidence.

Structured evidence committed under the shot's `logs/`: `r7_defect_audit.json`,
`r7_composite_manifest.json`, `r7_video_review.json`. Original rejected logs are preserved.
Videos and derived image caches follow the existing gitignore policy and remain on disk.

## 3. Diagnosis and rejected quick fixes

The neutral guard does not protect sampled teeth/sclera, but their estimated base colours
are tens of degrees from the illuminant, not barely outside its 1.8-degree cone. The local
opening is an invalid material reference in these regions. A wider guard is not a proven fix.
Kernel-width experiments prove sensitivity of the hair subtraction, but not literal
15x15 block footprints. The code's named radius is actually kernel width.

The catchlight cap lacks eyelid intersection. Default-off is intentional: adding a lid mask
would still leave incorrect source occlusion and an unsafe old-reflection removal stage.
Corneal renderer, illuminant measurement, face landmarks and eye geometry are retained.
The saved renderer validation checks pass at 0.44–3.51%; no fresh renderer validation is claimed.

## 4. Still broken and settled

Studio sheen, lip highlights and original eye catchlights remain in the photograph and cannot
move with the new light. The ratio still has round 4's geometry, source-detail and temporal
ceilings. No photorealism row is newly green. **Do not regenerate downstream estimates from
the rejected diffuse layer.** Do not enable either rejected stage merely because files exist.
Do not require on-set references, change key thresholds, or enable persistent Cycles data.

## 5. Starter prompt for round 8

> Read round7.md and round6.md's rejection addendum. Begin from the verified round-4-equivalent
> defaults. If investigating specular removal, prove preservation of teeth/sclera and hair
> detail plus temporal coherence before enabling it; do not recycle the morphological-opening
> split as established physics. If investigating catchlights, fix eyelid/source occlusion and
> removal of the original reflection together. Keep experiments separate and off by default.
> Produce and inspect complete comparison videos, record the actual playback/inspection method,
> and obtain the artist's verdict without claiming photorealism from statistics alone.
