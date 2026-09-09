# Round 29 handoff — sharper backing competes with light; no removal earned

**Date:** 2026-09-09. Starts from round 28. Production defaults unchanged.
**Delivered:** general SG radiance penalty, narrow-light/sampling/geometry
comparison, spatial backing competitor, reflection-balanced whole-frame folds,
independent scalar transport checks, exact replay/fallback and two videos.
**Not delivered:** clean iris or pupil material, catchlight removal, fresh final
validation, continuous inverse tracking, credible 3D lids, source-driven iris
illumination, actual scene replacement or photorealism.

## 1. Artist steering and scope

Stay on eyes; instruments remain deferred. Generic input support, observed
same-eye identity and exact keyed fallback remain required. No generated
performer texture, capture references, key retuning, tracking-only detour or
universal impossibility claim. Source eyes retain their original reflections
until the whole removal/actual-scene-replacement transaction earns support.

Read `docs/r29-eye-findings.md` and the frozen `docs/r29-eye-protocol.md`.
The numerical module accepts explicit observations and geometry. Reporter
frames, reused assistant annotations and six fixed geometry witnesses remain
diagnostic choices, not an automatic input-agnostic estimator. No fresh test
was consumed. Source 805 and 815 are seen.

## 2. What was tested

`eye_shape_fit.py` supports arbitrary SG vectors with the round-28 exposure-unit
full-sphere radiance penalty, cross terms and coefficient coercivity. The new
dictionary adds concentrations 128 and 512 at 24 deterministic farthest-point
centres drawn from fitting central reflected rays without RGB or labels.
Each held fold has its own training-only dictionary. Baseline/narrow dictionaries
have 53/101 light columns. Iris appearance is the existing 17x17 grid.

Cross six existing geometry witnesses with 3x3 and 7x7 native BOX quadrature,
and four arms: appearance-only, broad light, narrow light, narrow light with
17x17 spatial backing instead of constant backing. This yields 48 full models
and 192 held-fold fits per eye. Spatial backing is source-lit radiance, not
physical pupil material, and changes appearance prior dimension as well as
capacity. No source geometry or production transport was retuned.

Four leave-one-frame-out folds use source 730/760/790/820. Intersect all arms'
candidate, two-frame appearance and five-degree/all-node angular support.
Eligible folds need eight labelled reflections and eight other pixels; at
least two eligible folds are required for selection. Equal-weight reflection
and other-pixel MSE within each fold, then equal-weight eligible folds.

| Held source | Left supported reflection / other | Right supported reflection / other |
|---|---:|---:|
| 730 | 22 / 68 | 35 / 138 |
| 760 | 2 / 130 | 27 / 145 |
| 790 | 2 / 130 | 28 / 165 |
| 820 | 6 / 84 | 22 / 173 |

**Left abstains from selection; right uses all four folds.** This coverage was
measured before fitting; thresholds were not relaxed. The all-arm grid/BOX7
reachability intersection changes support relative to round 28, so comparisons
to its scores are not paired improvements. Both eyes' predictions are persisted
before either diagnostic 805/815 RGB read.

## 3. Independent and real findings

Independent scalar spherical intersection, Snell/Fresnel and pixel integration
agree with production on five signed positional cases to 2.23e-11. Independent
sphere quadrature agrees with the generalized Gram matrix to 4.85e-15.
Known-light native integration at BOX3/7/21 differs from scalar BOX41 by
.00416711/.000117108/.000117103 max linear. These are supplied spherical-eye
controls, not anatomical validation. The narrower fixture basis receives the
known true light axis; this is not automatic light-localization evidence.

Broad vs narrow held sum max error improves .0197942 to .00100541, while held
layer max error remains .0260851 vs .0223170. A single-view spatial appearance
alternative has an exactly identical sum with zero reflection and .0279707
layer difference; it is not a shared temporal material solution. Its initial
>.05 discrepancy assertion was corrected to >.01 after measuring the .02797
held reflection peak; no fixture light, geometry or real thresholds changed.

On real fitting catchlights, narrow models still have median display errors
43.776-52.487 left and 47.966-53.428 right over geometry/sampling configurations.
Spatial backing reduces those ranges to 26.231-32.445 and 20.127-26.240, but
does not make the explanation physical.

Right selects model 3: geometry witness 0, BOX3, spatial backing. The table
holds that geometry/BOX fixed across arms; left uses the same **unselected**
fixed diagnostic witness. Display errors are per-native-pixel RGB mean absolute
sRGB differences on a 255-level scale.

| Arm | Left fit catchlight median | Right fit catchlight median | Right balanced fold MSE | Left seen-815 catchlight median | Right seen-815 catchlight median |
|---|---:|---:|---:|---:|---:|
| Appearance only | 51.169 | 61.882 | .0334558 | 69.102 | 63.341 |
| Broad | 55.471 | 55.528 | .0287482 | 65.035 | 57.908 |
| Narrow | 50.677 | 48.883 | .0255824 | 69.042 | 62.875 |
| Narrow + spatial backing | 31.914 | 25.547 | **.0152352** | 47.827 | **74.557** |

Seen 815 support is 181/80 pixels left/right, including 16/38 labelled
reflections; right selected p95 reflection error is 157.444. Seen 805 support
is 41/204, including zero/35 reflections. Left spatial backing's lower 815
median hides a worse linear MSE (.0489275 vs broad .0301177); it is not a pass.

The visually sharper spot is largely painted into pupil backing. On right
source-730 reflection pixels, narrow constant backing/reflection means are
.003079/.125470 linear RGB; selected spatial backing/reflection means are
.200181/.009489. At 805 the spatial model assigns .251381 to backing and
.010735 to reflection. Its later seen prediction fails. These are retained
failed explanations, not credible clean-layer uncertainty bounds.

BOX3 coefficients evaluated at BOX7 change candidate sums by max .051818 left
and .031220 right across all arms; narrow-only values are .038787/.008622.
Refitting changes sums by up to .063780/.031689. Appearance-only models have
the largest differences, so the effect is not solely narrow-light aliasing.
BOX7 convergence on real footage and the actual camera PSF remain unmeasured.
All candidate model maxima stay below .711023 left/.801959 right; sampled
stability is not a bound on unobserved environment radiance.

## 4. Verification and review

All **96 full models/predictions and 384 held-fold scores replay exactly**.
Held-fold fitter RGB access is guarded; seen 805/815 fitting reads are zero.
Frozen source and round-27/28 artifacts are unchanged. **300 exact keyed-eye
fallbacks, 304 protected hashes and 12 disabled complete transactions pass.**
Final verifier exits zero. No production numerical module changed.

- `final/r29_shape.mp4`: 150 continuous frames, 60000/1001 fps, 768x960,
  2.5025 seconds. Source left, sparse sampling overlays right. Seven decoded
  contact sheets inspected, including both blink sequences. Original eyes and
  reflections persist. Maximum frame mean encode error 1.681764 display levels.
- `final/r29_layers.mp4`: 144 frames at 24 fps, 1000x580, six seconds of
  explicitly labelled one-second still holds. Matched source/broad/narrow/
  spatial-backing sums plus narrow reflected layer. All six decoded holds
  inspected. Black is unscored; zero displayed model pixels clip above white.
  Maximum frame mean encode error .415507 display levels.

Videos are display-sRGB CRF12 reviews, not HDR masters, live playback or artist
acceptance. Encoding means include black/unchanged pixels and are not repair
scores. Artifacts live in `composite/r29_shape`, `composite/r29_layers`, and
`logs/r29_*`. Raw command logs are `logs/r29/`. Existing AGENTS/round-17 work
and the generic latest review log are excluded from this round's commit.

```powershell
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r29_shape.py --shot music_videos/reporter_test/shots/reporter_a --preflight
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r29_fixture.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r29_shape.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/summarize_r29_shape.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/review_r29_shape.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r29_shape/review --fps shot --out r29_shape.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r29_layers/review --fps 24 --out r29_layers.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r29_shape.py --shot music_videos/reporter_test/shots/reporter_a
```

The verifier also supports `--skip-videos` followed by `--videos-only`; the
latter rechecks hashes, fixture and fallback while retaining completed fit
replay results. This was used during encoding to avoid repeating the full fit
suite when only the review was added.

## 5. Starter prompt for round 30

> Stay on eyes. Read round29.md and docs/r29-eye-findings.md. Narrower light
> helps a known-axis scalar fixture's sum but leaves wrong layers. On real
> reporter observations, narrower light plus six fixed geometry witnesses still
> fails training catchlights; a spatial backing paints a sharper photographed
> spot below the cornea, wins right reflection-balanced folds, then fails seen
> 815 (median 74.557 levels, p95 157.444). Left has only one eligible fold and
> correctly abstains. Preserve these failures and do not call a sharper sum
> clean material. BOX3 vs BOX7 materially changes predictions, particularly
> appearance boundaries; real sampling convergence and camera PSF are unresolved.
> Before another fresh final check, advance coupled source geometry/light/material
> inference with explicit competition against spatial backing. Diagnose whether
> continuous optical pose/light localization and source sampling can explain
> the observed compact reflection consistently across time. Keep source-driven
> iris illumination, credible 3D lid/light visibility and actual scene-light
> replacement in scope; stationary source-lit grids cannot complete that task.
> Keep full-sphere radiance regularization, reflection-covered evaluation,
> explicit angular abstention and exact keyed fallback. Source 805/815 are seen.
> No generated performer texture, key retuning, capture requirements,
> tracking-only detour or universal impossibility claim. Produce a temporal
> review and round30.md with measured findings and remaining physical work.
