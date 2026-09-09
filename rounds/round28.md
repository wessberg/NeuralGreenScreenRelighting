# Round 28 handoff — stable radiance, pupil-covered catchlight prediction fails

**Date:** 2026-09-09. Starts from round 27. Production defaults unchanged.
**Delivered:** exposure-unit spherical radiance regularization, angular sampling
checks, controlled extrapolation/ambiguity fixtures, decentered pupil/backing
transport, real reflection-covered fitting/scoring, exact replay/fallback and
two encoded reviews.
**Not delivered:** clean iris material, catchlight removal, fresh held-source
validation, continuous inverse tracking, credible 3D lids, physical iris-light
coupling, actual scene-light replacement or photorealism.

## 1. Artist steering and scope

Stay on eyes; instrument work deferred. Generic input support and observed
same-eye identity remain required. Keep keyed eyes and original catchlights
until the complete source-removal/actual-scene-replacement transaction earns
support. No generated performer texture, key retuning, capture references,
tracking-only detour or universal impossibility claim.

Read `docs/r28-eye-findings.md` and the frozen `docs/r28-eye-protocol.md`.
The numerical module takes explicit observations and geometry; fixed reporter
frames, assistant annotations and six reused witnesses are diagnostic choices,
not an automatic input-agnostic production estimator. Camera and scale remain
assumed. No neural model or Cycles render ran.

Round 27's starter prioritized stability before another final check. This round
completed those prerequisites and added real reflection evidence. **No fresh
test was spent:** the pupil-covered model already fails catchlights on fitting
observations. Source 815 remains seen. The downstream physical solve and scene
coupling remain open, rather than silently being dropped from the task.

## 2. Radiance stability and controlled evidence

`eye_radiance_fit.py` replaces sampled-column ridge with mean pixel residual
plus 1e-3 mean squared appearance coefficients and a penalty on full-sphere mean
squared reflected radiance. The SG Gram integral is analytic, includes cross
terms, and matches independent quadrature to 2.11e-15. A 1e-6 coefficient term
makes the light penalty positive definite. Fixed reflected weights are 1e-6,
1e-4, 1e-2, plus an appearance-only arm. These are declared working-exposure
priors, not measured SI radiance. Column preconditioning transforms the penalty.

An independent scalar fixture with separately varied Fresnel/material/direction
has withheld sum/layer max linear errors .000266171/.000277567. It deliberately
provides excitation; it is not an anatomical renderer or a real-eye proof.
Duplicate observations and sevenfold exposure scaling change coefficients by
1.67e-15/5.33e-15 relative to their expected solutions.

The tiny-tail adversary falls from 2,181,619.86365 maximum predicted linear
radiance to .119999568. A distinct known withheld bright SG lobe is still
predicted badly (max error .478919483) and is rejected by angular sampling.
The gate accepts 10/10 covered queries and rejects all ten of each negative
case: unseen directions, one frame repeated, or one bad BOX node.

**Angular coverage is not layer observability:** two positive explanations,
`A=.08,L=1` and `A=.1,L=.02`, both with `F=.02`, have identical sums but
transmitted contributions differing by .0196 linear. Regularization cannot
turn its preferred explanation into clean material.

On the **original round-27 matrices**, all 24 new-objective models per eye have
maximum 815 predictions .160059 left/.251808 right, versus the frozen old
right family's 201,740.159504. This is a same-support numerical-stability
diagnostic, not a repair result or an isolated light-penalty ablation: the
appearance regularizer changed too. Frozen round-27 results are preserved.

## 3. Pupil transport and real reflection-covered failure

`trace_connected` has an optional decentered pupil aperture. Refracted rays
must reach iris or recessed backing before the globe wall. Default centered
behavior replays round 27 exactly. Four signed offset controls each change
32 classified pixels; 1,296 closed samples reject. Independent scalar BOX
integration matches to 2.22e-16, with a genuinely mixed iris/backing pixel.
Initial test coordinates were pure surfaces; inspection caught and fixed the
test coverage without changing source fitting or thresholds.

The model fits unknown nonnegative constant backing RGB shared across four
fitting observations, a 17x17 source-lit iris grid and the existing 53-column
light field. BOX footprints may mix backing/iris. Reflected visibility remains
assumed open, and the inherited oversized shell does not establish credible
3D lids. The appearance grid is not albedo or coupled to fitted iris irradiance.

Fit source 730/760/790/820, select on seen 805, diagnose seen 815. Reuse six
round-27 boundary witnesses per eye without retuning geometry. All comparisons
use support common to all witnesses. Appearance coefficients need two fitting
frames; every reflected BOX node needs a direction within five degrees in two
distinct frames. This is a fixed sampling criterion, not confidence.

| Support | Left | Right |
|---|---:|---:|
| Four-frame fit native pixels / labelled reflection pixels | 1,053 / 122 | 1,048 / 139 |
| 805 candidate / jointly supported pixels | 276 / 45 | 304 / 214 |
| 805 jointly supported labelled reflections | **0** | 38 |
| 815 candidate / appearance-supported / angular-supported | 319 / 268 / 207 | 314 / 279 / 131 |
| 815 jointly supported pixels / labelled reflections | **207 / 16** | **115 / 41** |

Round 27 included eight/two labelled reflection fitting pixels and zero at 815.
Backing now gets reflection-covered evidence into the fit, but incomplete masks
and assumed geometry remain. The new support is not comparable to the old
annulus-only error as a round-over-round quality gain.

| Selected on 805 | Left | Right |
|---|---:|---:|
| Model / geometry witness | 0 / 0 | 1 / 0 |
| Reflected weight | none | 1e-6 |
| Fit / development linear MSE | .007884762 / .000148006 | .008336739 / .009302121 |
| Seen 815 linear MSE | .006464097 | .025466827 |
| 815 all-supported median / p95 display error | 10.3661 / 90.6366 | 23.1435 / 110.2382 |
| 815 labelled reflection median / p95 display error | **69.1028 / 93.0343** | **52.2301 / 123.5674** |

The left development support has zero labelled catchlights, so its small
development score cannot validate reflection quality. The right reflected
choice beats its appearance control on 805 MSE (.00930212 vs .01067911), but
loses on seen-815 MSE (.02546683 vs .02369824). No selection was changed on 815.
Both eyes' predictions were saved before either 815 RGB was read by scoring.

Training labelled-reflection median display errors are already 38.0372–65.7335
left and 48.6414–62.4116 right across four fitting frames. This is now a source
fit failure, not only extrapolation. The current diagnostic does not isolate
light shape, geometry, PSF, visibility or backing as the single cause.

The review makes the failure obvious: the left appearance model makes a flat
grey pupil; the right reflected model makes a broad glow rather than the
photographed compact catchlight. Left backing RGB is (.175902,.151071,.142655);
right selected backing is (.012647,.013084,.020416). The right appearance-only
control instead uses a bright flat backing. These are retained failed
explanations, not credible clean-layer alternatives. No replacement is earned.

## 4. Verification and review

All 48 new fits/designs/predictions/selections and all 48 regularized fits on
old support replay exactly. All 48 frozen round-27 fits/designs/predictions and
controls remain exact. Two target-RGB substitution/read-guard runs per eye make
zero 815 reads during fitting and preserve predictions. Source/artifact hashes
match. **300 exact keyed-eye fallbacks, 304 protected hashes and 12 disabled
complete transactions pass.** Final verifier exits zero.

- `final/r28_radiance.mp4`: 150 continuous frames, 60000/1001 fps, 768x960,
  2.5025s. Source left, sparse sampling overlays right; original eyes/catchlights
  persist through motion and blinks. Seven decoded contact sheets and decoded
  815 inspected. Maximum frame mean encode error 1.712402 display levels.
- `final/r28_layers.mp4`: 72 frames, 24 fps, 960x636, three seconds of explicitly
  labelled one-second holds, not motion. Selected / appearance-only / best
  reflected models show source, sum, appearance/backing and reflection on
  identical support. All three decoded holds inspected; zero displayed model
  predictions clip above white. Black means unscored. Maximum mean encode error
  .235328 display levels.

Reviews are display-sRGB CRF12, not HDR masters, live playback or artist
acceptance. Encoding-error means include unchanged/black pixels and are not
reconstruction scores. Exact-source claims concern linear arrays, not encodes.
Artifacts are under `composite/r28_radiance` and `composite/r28_layers`; logs
are `logs/r28_*`. Existing AGENTS/round-17 work and the generic latest review
log are excluded from this round's commit.

```powershell
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r28_radiance.py --shot music_videos/reporter_test/shots/reporter_a --freeze-only
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r28_fixture.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r28_radiance.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/review_r28_radiance.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r28_radiance/review --fps shot --out r28_radiance.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r28_layers/review --fps 24 --out r28_layers.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r28_radiance.py --shot music_videos/reporter_test/shots/reporter_a
```

## 5. Starter prompt for round 29

> Stay on eyes. Read round28.md and docs/r28-eye-findings.md. Radiance-unit
> regularization removes the sampled-tail blow-up, and pupil/backing transport
> gets 122/139 labelled reflection pixels into source fitting and 16/41 into
> seen-815 scoring. Yet the selected models already fail training catchlights
> by tens of display levels. A flat bright backing or broad reflected glow
> is not clean material or removal. Preserve these failed witnesses. Before
> spending a fresh final check, diagnose source reflection shape jointly with
> optical geometry and camera sampling; test any richer/narrower light model
> against independent controlled transport and competing backing/material
> explanations. Keep the radiance penalty and explicit angular abstentions,
> but do not mistake five-degree sampling proximity for layer observability.
> Left development support excludes all labelled catchlights; fix evaluation
> design prospectively if choosing reflection models. Move toward coupled
> source material/geometry/light inference, credible 3D lid visibility and actual
> scene-light replacement; a sharper fitted photographed sum alone is not a
> layer split. Source 815 is seen. Preserve exact keyed source until the whole
> removal/replacement transaction earns support. No generated performer texture,
> key retuning, capture requirements, tracking-only detour or impossibility claim.
> Produce a temporal review and round29.md with measured findings and limits.
