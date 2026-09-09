# Round 30 handoff — local pose/light fit exposes solver failures; eyes unchanged

**Date:** 2026-09-09. Starts from round 29. Production defaults unchanged.
**Delivered:** continuous shared optical corrections and source-light fitting,
spatial-backing competition, four whole-frame folds, denser sampling diagnostics,
independent scalar controls, exact replay/fallback checks and two review videos.
**Not delivered:** validated source geometry, clean iris/pupil material, catchlight
removal, fresh final validation, continuous inverse tracking, source-driven iris
illumination, credible 3D lids or actual scene replacement. Photorealism is not met.

## 1. Artist steering and scope

Stay on eyes; instruments remain deferred. Generic input support, observed
same-eye identity and exact keyed fallback remain required. No generated
performer texture, capture references, key retuning, tracking-only detour or
universal impossibility claim. Original eyes and reflections remain until the
complete removal/actual-scene-replacement transaction earns support.

Read `docs/r30-eye-findings.md` and frozen `docs/r30-eye-protocol.md`.
The numerical module accepts explicit observations and geometry; the reporter
adapter still uses reused annotations and supplied boundary geometry. It is
not an automatic input-agnostic estimator. Source 805 and 815 remain seen.

## 2. What was tested

`eye_continuous_fit.py` adds one shared optical correction per eye: projected
globe x/y translation bounded to one native pixel each and yaw/pitch bounded
to five degrees each. Corrections transfer to held frames without held RGB.
Use round-27 geometry witness 0 and round-29 native candidate coordinates.
Keep the 53 broad SG columns and add one lobe seeded at the brightest fitting
sample's reflected central ray. Fit tangent offsets within +/-20 degrees each
and concentration 128..8192 (initially 512).

Four arms: fixed seed; optimized light; optimized light plus shared optical
correction; and the latter with 17x17 spatial pupil backing. Iris remains a
17x17 source-lit grid. Preserve nonnegative fitting, full-sphere radiance
regularization/cross terms and coefficient coercivity. Add 1e-4 squared
normalized geometry prior. Four alternating sweeps, each at most 16 nonlinear
evaluations, use BOX7; only descending complete objectives are accepted.
Every original training RGB stays in the loss; invalid transport is penalized.

There are **eight full fits and 32 held-fold fits**. Folds hold source
730/760/790/820 individually. Every seed and fit is training-only. Intersect
each fitted arm's valid transport, two-frame appearance occupancy and
five-degree/all-node angular support with the frozen round-29 masks.

| Held source | Left supported reflection / other | Right supported reflection / other |
|---|---:|---:|
| 730 | 22 / 68 | 35 / 136 |
| 760 | 2 / 130 | 27 / 145 |
| 790 | 2 / 130 | 28 / 165 |
| 820 | 6 / 83 | 22 / 173 |

Eight reflections/eight others per eligible fold and two eligible folds are
still required. **Left abstains; right uses all four folds.** Relative to the
old mask, left loses one pixel at 820/815 each; right loses two at 730 and one
at 805. No reflection counts change. Support is partly adaptive to fitted
geometry, so these are not paired improvements over round 29.

## 3. Measured findings

All round-30 arms below score identical native pixels. Display errors are
per-pixel mean RGB absolute sRGB differences on a 255-level scale.

| Arm | Left fit reflection median | Right fit reflection median | Right balanced fold MSE | Left seen-815 reflection median | Right seen-815 reflection median |
|---|---:|---:|---:|---:|---:|
| Fixed seed | 53.029 | 48.747 | .0264951 | 63.909 | 58.429 |
| Light fit | 43.994 | 39.314 | .0271758 | 59.278 | 61.620 |
| Pose + light | 53.035 | 44.601 | .0278678 | 63.907 | 60.288 |
| Spatial backing | 31.709 | 24.857 | **.0150917** | 48.487 | **77.695** |

Right selects spatial backing, then fails seen 815 with reflection p95
**159.587** and linear MSE **.1002623** (fixed: .0703360). Left spatial
backing's lower median again hides worse MSE (.0468185 vs .0359714 fixed).
Seen 815 has 180/80 supported pixels left/right, including 16/38 reflections.
Seen 805 has 41/203, including zero/35 reflections.

On right fitting reflections, coupled constant backing/reflection means are
.012664/.165573 linear RGB; spatial backing/reflection means are
.231339/.012287. Left spatial backing/reflection means are .235999/.004457.
The sharper spot is mainly in backing. It is not recovered clean material.

The local optimizer is itself inadequate: **66/120 nonlinear sweeps exhaust
their evaluation limit**. All accepted objectives descend; zero proposals are
rejected. Some coupled fits retain one invalid training pixel with penalty.
Both full light-only models reach the broad lobe bound (128.3554 after the
Jacobian's bound margin). Full right coupled translation is (-.0792,.1895)
pixels with yaw/pitch changes (.0263,.1183) degrees; full left barely moves.
The coupled arms start from the original fixed seed, not the optimized
light-only solution. More degrees of freedom do not guarantee a better local
result. The sampled aperture objective is piecewise smooth; derivative
convergence and its role in stagnation have not been measured.

**Independent fixture:** compact vs dense transport agrees to 4.44e-16 on
eight signed cases; independent scalar direction/Fresnel agrees to 2.70e-11
and integrated RGB to 7.49e-11. Unknown-axis fitting fails: its brightest-RGB
seed is **33.681 degrees** from true light, outside the tangent search's
approximately 27.24-degree maximum reach. The localized lobe receives exactly
zero RGB coefficients in all arms; holding those coefficients fixed gives
its nonlinear move no signal. Fixed/light held sum/layer maxima are
.0246736/.0478343; coupled .0247279/.0478977; spatial .0475209/.0540543.
No known axis was supplied and no test RGB rescued the fit. This is a solver
initialization/inactive-component failure, not a universal material limitation.

**Sampling:** unchanged full coefficients at BOX3/7/15 compared with BOX31
give across-arm maximum sum differences of .0304188/.0057624/.0029771 left
and .0225951/.0038717/.0015595 right. Material boundary components can cancel:
spatial backing changes by up to .0093116/.0057253 at BOX7 vs31. Per-arm
support differs; no paired arm ranking or measured camera PSF follows.
BOX31 is a denser reference, not a proof of convergence. Original left
source-815 pupil boundary error remains about 2.77 pixels.

## 4. Verification and review

All **40 final coefficient solves, forward predictions and support masks
replay exactly**. Two complete nonlinear source-730 held-fold trajectories
(one per eye) also replay exactly; the other 38 trajectories are not rerun.
Held RGB is guarded; seen 805/815 fitting reads are zero. Both eyes' full/fold
predictions were persisted before seen scoring. Frozen previous evidence is
unchanged. **300 exact keyed-eye fallbacks, 304 protected hashes and 12
disabled complete transactions pass.** No production module was modified.

- `final/r30_continuous.mp4`: 150 frames, 60000/1001 fps, 768x960, 2.5025 s.
  Source left, sparse sampling overlays right. All seven decoded contact
  sheets inspected, including both blink sequences. No continuous repaired
  eye is shown. Maximum frame mean encode error 1.686323 display levels.
- `final/r30_layers.mp4`: 144 frames at 24 fps, 1200x580, six labelled
  one-second still holds. Observed source, four model sums and coupled
  reflected layer. All six decoded holds inspected, including the wrong
  source-815 spot. Black is unscored; no displayed model pixels clip above
  white. Maximum frame mean encode error .407545 display levels.

These are display-sRGB CRF12 reviews, not HDR masters, live playback or artist
acceptance. Encoding averages include unchanged/black pixels, not repair scores.
Artifacts are in `composite/r30_continuous`, `composite/r30_layers` and
`logs/r30_*`; raw command logs are `logs/r30-*.log`. Existing AGENTS/round-17
work and the generic latest review log are excluded from this round's commit.

```powershell
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r30_continuous.py --shot music_videos/reporter_test/shots/reporter_a --preflight
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r30_continuous.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/summarize_r30_continuous.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r30_continuous.py --shot music_videos/reporter_test/shots/reporter_a --skip-videos
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/review_r30_continuous.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r30_continuous/review --fps shot --out r30_continuous.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r30_layers/review --fps 24 --out r30_layers.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r30_continuous.py --shot music_videos/reporter_test/shots/reporter_a --videos-only
```

The audit supports `--resume` only with matching saved fitter/adapter hashes.
The summary's first run failed because the adapter's `label` holds a boolean
reflection mask, not the original annotation dictionary. The summary now
loads the source annotations explicitly; this did not change any fit or gate.

## 5. Starter prompt for round 31

> Stay on eyes. Read round30.md and docs/r30-eye-findings.md. Local continuous
> optical/light fitting still fails source sums and layers; spatial backing
> wins right folds then fails seen 815 (median 77.695, p95 159.587). Left still
> abstains. Do not interpret this local solver as a physical impossibility:
> the independent fixture initializes 33.681 degrees from true light, outside
> its search, assigns the lobe zero energy and cannot activate it during a
> fixed-coefficient light move. Real fits exhaust 66/120 nonlinear budgets.
> Before a fresh final check, build and independently test training-only
> source-light initialization/activation with enough directional reach and
> optimization that can escape inactive lobes. Challenge it with spatial
> backing, multiple starts and known sum/layer controls without passing true
> axes into the real solver. Measure material-boundary sampling/derivative
> behavior rather than treating a denser BOX reference as a measured PSF.
> Keep coupled source geometry/light/material inference, source-driven iris
> illumination, credible 3D lid/light visibility and actual scene replacement
> in scope; stationary source-lit grids cannot complete that task. Preserve
> full-sphere radiance regularization, reflection-covered whole-frame scoring,
> explicit angular abstention and exact keyed fallback. Sources 805/815 are
> seen. No generated performer texture, capture requirements, key retuning,
> tracking-only detour or universal impossibility claim. Produce a temporal
> review and round31.md with measured findings and remaining physical work.
