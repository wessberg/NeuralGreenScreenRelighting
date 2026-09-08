# Round 12 handoff — native eye observations, failed correspondence, neural prior art

**Date:** 2026-09-08. Starts from round 11 and unchanged production defaults.
**No real reflection removal or replacement was delivered.** This round adds a
reusable observation stage and tests a native boundary/correspondence hypothesis on
real footage. It does not satisfy the original round-12 goal of trustworthy real
repairs. The next direction is to evaluate existing neural components before more
custom detector development. Original catchlights, crown sheen and lip gloss remain.

## 1. Artist decisions and corrections — preserve these

- The artist reiterated that this must be **generic and reusable on any input
  footage**, not a tailored reporter correction. Generic interfaces do not prove
  generalization; label the limits of the actual evaluation.
- The artist asked why we were rebuilding this instead of using existing neural
  solutions. Relevant prior art does exist. I should have evaluated that landscape
  before investing in another handcrafted detector. Do not repeat that ordering.
- The task is **not removal alone**. We must track the eye and add reflections
  generated from the scene's lights. Segmentation, pupil/iris motion, corneal pose,
  lid/lash/brow visibility and source occlusion remain necessary even if a neural
  model supplies the removal stage.
- Round 11's accepted fallback persists: when detection, angle, correspondence or
  evidence is unreliable, keep the exact keyed eye and its original catchlight.
  Couple removal and scene-reflection replacement; never add a new reflection over
  a failed removal. Use the actual corneal renderer, not a tracked painted dot.

## 2. Delivered reusable code and boundaries

`scripts/iris_observations.py --shot <dir> --review`:

- Consumes the shot's **scene-linear Rec.709 keyed EXRs** and existing face landmark
  sidecar. It contains no reporter identity, frame range, eye coordinates or camera
  constants. Requires the key/track pixel spaces to agree explicitly.
- Derives native ROI sizes from measured eye scale, approximate roll from the eye
  region's principal axis, topology and pixel aspect from the sidecar. Landmark
  polygons supply no semantic material labels. Native fitting samples physical
  circles through the declared pixel-aspect mapping, without resampling its input.
- Writes per-eye RGB/mask NPZs and `relight/iris_observations/manifest.json`.
  Geometry is ROI-local; origin, seed, radius, roll and coordinate metadata are
  explicit. `logs/iris_observations.json` is the compact summary.
- Uses `temporal_iris.apply_eye_repair` for exact source fallback. All geometry and
  correspondence validation flags, and `scene_reflection_allowed`, remain false.
  **This is an observation/fallback stage, not a working automatic remover.**

`scripts/native_iris.py` is an experimental native boundary fitter. It searches
lateral/lower pupil and limbus arcs using six sectors of radial contrast, excludes
bright contributors before pupil fitting, rejects unresolved/nonconcentric fits
and search-boundary solutions, and proposes a conservative lower-iris interior.
The near-circular model after pixel-aspect correction does not resolve oblique
refraction, true semantic occlusion or arbitrary pupil deformation. Fixed contrast
thresholds are hypotheses, not universal detection guarantees. Top iris/lids are
not labelled as clean material. Reflection candidates are **not removal masks**;
some still include rim/skin pixels.

`scripts/iris_correspondence.py` fits a small relative torsion on one spatial sector
and validates on a separated sector. Donor training/validation bilinear footprints
cannot overlap. It has a wrong-rotation control and uses the shared material sampler.
The target is scrubbed before native detection, matching, ranking and confidence.
These checks do not certify unseen material under reflections.

`audit_r12_native.py` is explicitly a **reporter-only evaluation adapter** around
round 10's native cache. Its 64-pixel diagnostic crop, source frames and manual labels
do not appear in the reusable observation stage. `verify_r12.py` records sparse
semantic review, coordinate/fallback contracts, protected hashes and video decoding.

`temporal_iris.py`, `composite_ratio.py`, original keys, source and shot config were
not changed. No model packages were installed and no neural inference ran this round.

## 3. Real-frame measurements — insufficient for repair

On **1,200 original source frames per eye**, final native fitter candidates:

| | Subject left | Subject right |
|---|---:|---:|
| Numerical candidates | 149 | 92 |
| Candidate reflection exclusion overlaps proposed pupil | 1,006 | 1,006 |

The pupil-overlap count includes rejected observations and possibly false bright
masks; it is not a ground-truth glint count. Pilot counts changed during development;
`r12_native_audit.json` contains the final counts, `r12_pilot.json` only the last pilot.

Sparse native annotations were made on source **79, 217, 521, 779, 923, 1115**, both
eyes, before viewing their fitted overlays in that evaluation pass. They are
assistant annotations, **not artist or independent human ground truth**, with
estimated boundary uncertainty 1.5 native pixels. Of 12 eye-frames:

- All four manually rejected blink/closed views were rejected.
- Only **one of eight open-eye views** passed the numerical candidate gate.
- Candidate material masks admitted **26/40** sparse clean-iris points and **0/60**
  sparse noniris points. This sparse result does not certify dense visibility.
- Individual eye-frame pupil boundary-distance p95 reaches **5.34 px**, limbus
  p95 **7.08 px**. Some rejected images visibly produce tiny false pupil circles.
  One accepted open view has pupil p95 **2.31 px**, already substantial at this scale.

The correspondence test checks every selected frame (source 700–849) with two fixed
seed-relative potential holes. Each hole is **five native pixels**, eligible only
when the preliminary material mask admits all five. Just **14 cases / 70 requested
pixels** qualify. This is conditional, sparse evidence and a smaller hole than many
actual reflections; it must not be marketed as representative recovery accuracy.

| Matched supported subset | Fixed coordinates | Fitted torsion |
|---|---:|---:|
| Cases / pixels | 12 / 54 | 12 / 54 |
| Per-case display MAE median | 3.2601 | 3.6259 |
| Per-case display MAE p95 | 6.2398 | 7.1897 |
| Relative RGB error median | 4.44% | 4.99% |
| Relative RGB error p95 | 11.25% | 11.81% |
| Pixels passing context + donor-agreement gates | **0** | **0** |

Display errors use 0–255 channel units on decoded native source crops; these are
not scene-linear thresholds. The fixed arm supports 55 pixels overall, torsion 56;
the paired table uses their shared 54. The extra torsion fit does not improve this
test. No photometric gain/colour correction is fitted.

All **14 hidden-RGB invariance cases**, each using two extreme perturbations, pass;
12 have nonzero donor support. Geometry, masks used for matching, donor decisions
and predictions are invariant after scrubbing. An unguarded detector centre changes
in **3/14** cases, demonstrating why freezing a detector computed from hidden target
pixels is not a valid leakage control. This does **not** test the upstream face
tracker: coarse ROI/seed are fixed, and eligibility is conditioned on a preliminary
mask. Do not describe it as end-to-end detector-independent ground truth.

The generic stage separately ran on **150 keyed frames**, yielding 58 numerical
eye candidates and **zero changed channels** in all 300 fallback eye crops. These
counts differ from the diagnostic because keyed colour, dynamic ROIs and roll differ.
No repair/fallback transitions exist in this output because all eyes retain source.

## 4. Verification and review artefacts

`verify_r12.py` passes:

- All **304 protected-file hashes** still match round 10's detailed manifest.
- Missing detection returns fallback.
- Dynamic native ROI and exact fallback contracts pass at two image scales and
  with the anamorphic pixel-aspect mapping. These use geometrically transformed
  **real** pixels; they are adapter tests, not additional footage or positive repair
  evidence. No second prepared/tracked clip exists in this workspace.
- A masked native detector test also passes under pixel-aspect correction and roll.
- All five new Python files compile; `--shot` omission lists available shots.

Review-only CRF 12 videos in the shot's `final/`:

- `r12_native_source_fallback.mp4`: **150 frames at source 60000/1001 fps**. Original
  keyed eyes left, exact fallback right. Both eyes shown. All 150 ordered thumbnails
  inspected in two aspect-preserving sheets. Original glints and blinks are retained.
  This is preservation evidence, not a visible improvement.
- `r12_native_masks.mp4`: **1,200 frames at source fps**, native source beside
  candidate boundary/material overlays. Full source 779 inspected after final fit;
  source survey, pilot and development counterexamples were also inspected. Not all
  final mask-video frames were visually inspected. The entire video was decoded.
- `r12_native_holdout.mp4`: **14 cases at 2 fps**, a labelled diagnostic slideshow,
  not a motion/flicker test. Full first case inspected; all cases decoded. Unavailable
  pixels are purple. Actual source reflections are untouched in every panel.

Total decoded: **1,364 frames**. This was image/contact-sheet inspection, not
real-time playback or artist approval. Re-running verification resets visual review
to pending. Detailed regenerable observations live under `composite/r12_native/`.

```powershell
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/iris_observations.py --shot music_videos/reporter_test/shots/reporter_a --review
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/audit_r12_native.py --shot music_videos/reporter_test/shots/reporter_a --evaluate
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name iris_observations_review --fps shot --out r12_native_source_fallback.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r12_native/masks --fps shot --out r12_native_masks.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r12_native/holdout --fps 2 --out r12_native_holdout.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/verify_r12.py --shot music_videos/reporter_test/shots/reporter_a
```

## 5. Prior art and direction change

See [docs/iris-prior-art.md](../docs/iris-prior-art.md) for primary sources and component
boundaries. Useful candidates found:

- **EllSeg / RITnet**: neural pupil/iris segmentation and tracking. EllSeg links
  pretrained checkpoints and eye-video inference. Evaluate this before extending
  native contrast heuristics. Full inferred ellipses are not visible-material masks.
- **ProPainter**: released video inpainting weights and temporal propagation. Worth
  a diagnostic comparison, but its output may reconstruct rather than copy texture,
  and the repo declares noncommercial use. It is not a drop-in production dependency.
- **TSHRNet**: released general RGB highlight-removal models; not specifically
  validated eye tracking, iris preservation or temporal reflection relighting.
- **Seeing the World through Your Eyes, CVPR 2024**: joint corneal-pose and iris/
  reflection decomposition, with public code linked. Especially relevant prior art,
  but its objective is the reflected scene; radial texture regularization is not
  evidence of exact performer-iris restoration.
- **EyePreserve / DeformIrisNet**: learned iris deformation with biometric identity
  objectives. EyePreserve's documented input is near-infrared plus a target mask;
  it is not an RGB reflection-removal service.
- **UnReflectAnything**: relevant recent highlight-removal work, but its README
  currently states that official pretrained weights are not released. The package
  API example must not be mistaken for available trained inference.

None was downloaded/run. Do not claim a ready solution was found or that none exists.
Use neural inference for geometry/masks/correspondence while keeping final performer
pixels tied to observed evidence. Existing `eye_geometry.py` and corneal Cycles work
remain valuable; flat-cap/lid/source-occlusion problems still block enabling the
previously rejected synthetic reflection composite. Hair/lips received no treatment.

## 6. Starter prompt for round 13

> Read round12.md and docs/iris-prior-art.md. The artist wants a generic pipeline that
> tracks each eye, removes the original corneal reflection where evidence supports
> it, and renders a replacement from actual scene lights. Unreliable views keep the
> original keyed eye/catchlight. They specifically challenged reinventing available
> neural solutions. Evaluate released pretrained eye segmentation/tracking first
> (e.g. EllSeg/RITnet), using an isolated compatible environment and checking actual
> weights availability. Compare on native real images, visibility and pupil boundaries;
> do not infer material visibility from a full ellipse. Keep the new observation API
> reusable and replace its experimental fitter as appropriate. Native/torsion tests
> yielded zero confidence and do not justify removal. Preserve hidden-footprint and
> detector-leakage controls; don't tune on the recorded sparse evaluation labels and
> then call them held out. Benchmark neural removal/propagation where useful, without
> silently promoting generated texture into the production performer. Retain corneal
> pose tracking and solve lid/lash/brow/source occlusion before enabling the Cycles
> reflection composite. Couple removal/replacement and fallback per eye. Produce a
> real source-versus-result video if any reliable repairs are earned. Do not call a
> detector or fallback video a reflection fix. Write round13.md for round 14.
