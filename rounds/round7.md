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

---

## 6. Supplement — the artist's original asks, and where they stand

Added by the Claude session that wrote the rejected round 5, after Codex's review. Section 5's
starter prompt stands; this records what round 5 was *for*, since none of it was achieved.

He opened the round with three requests. All three are **exactly where they were at the end of
round 4**:

1. **Eye catchlights**, explicitly noting the green-screen room had almost certainly already put
   one there. It had: a rectangular softbox reflection at 11 o'clock on both irises. It is still
   in the delivered composite, unmoved, because a smooth `R` can only recolour it.
2. **The top-lit hair sheen.** Still in the plate. Worth knowing: round 4 measures *worse* than
   the plate on this (sheen contrast 12.37 against the plate's 10.09) because its deeper
   terminators darken everything around a band it cannot touch. That figure is from
   `measure_specular_transfer.py` and was not disputed by the audit.
3. **Lips reading as "lit up more than the skin"** in round 4, compared with round 3. Measured on
   frame 21 with the comparison panel registered to correlation 1.0000, against a photographed
   lip-to-skin luminance of 0.8189: round 3 **0.9936** (+21.3 %), round 4 **0.8764** (+7.0 %). So
   round 3 was *further* from the photograph, not closer — what made it read better is that its
   flatter contrast hid the surviving gloss band. **Round 4's geometry fix is what exposed the
   specular problem**, which is worth remembering before "improving" contrast again.

### Two things measured in round 5 that survive its rejection

* **The E_source / E_target bandwidth gap is real and is NOT what lifted the lips.** `R = E_t/E_s`
  divides a smooth numerator (the proxy) by a sharp one (the neural normals): `E_source` carries
  3.75× more high-pass energy at 2 px, 1.81× at 8 px, crossing at ~32 px. Band-limiting it made
  the lip metric *worse* (+11.5 % vs +7.0 %). `--source-detail-sigma` ships off. Do not re-run
  this experiment without reading §2 of `round6.md`.
* **The plate's own key direction, from its own catchlights: 54.6° off the camera axis**, spread
  18.5°. The position does **not** come out — two corneas 63 mm apart cannot triangulate a source
  a metre away, and the least-squares point collapses onto her own head *with a small residual*,
  so a residual test passes on nonsense. Gated on reach-over-baseline. This matters beyond the
  eyes because `estimate_source_light.py` fits SH to a surface where every normal has `ny < 0`
  and its own log warns the direction is an extrapolation.

### Disk left behind, all regenerable

| path | size | keep? |
|---|---|---|
| `key/specular/` | 107 MB | rejected stage's output; unused by the default path |
| `composite/ratio_r5/` | 108 MB | the rejected look — **the only record of what he saw** |
| `composite/ratio_r7/` | 109 MB | the verified rollback; identical to `composite/ratio` |
| `composite/catchlight_render/` | 4.8 MB | 25 min of Blender; cheap to keep |
| `composite/r5_test/`, `composite/panels/`, `composite/r5_defects/` | small | `r5_defects/` holds his four screenshots and the measured neutral-loss note |

Shot is 3.6 GB. `composite/ratio_r7` is redundant with `composite/ratio` (bit-identical) and is
the obvious deletion if space is wanted.

### A methodological note, because it is the third instance

Round 4's despill bug was confirmed by three metrics that all computed the buggy gate. Round 5's
catchlight direction looked mirrored because the measurement inherited the stress rig's fill
light. And round 5 shipped because its author measured hair, iris and lips — and **not teeth,
sclera, or any temporal quantity** — then wrote it up from one still without watching the videos.
`audit_r5_defects.py` is the missing test, and it is anatomical rather than aesthetic on purpose.
**A round is not judged until someone has watched it move.**
