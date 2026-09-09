# Local pose/light fitting still does not earn source removal

Round 30 adds bounded continuous source-light and shared optical corrections to
the supplied-eye diagnostic. **No clean iris/pupil material, reflection removal
or actual scene replacement is earned.** Light-only fitting lowers training
catchlight error, but does not improve the right-eye reflection-balanced held
score. Spatial backing again wins that score and fails seen source 815.

The independent fixture also fails unknown-axis localization. This exposes
initialization and local optimization limitations; it is not evidence that the
observed reflection cannot be explained by a better physical solve.

## Model and actual scope

The frozen design is `docs/r30-eye-protocol.md`. The new numerical module,
`scripts/eye_continuous_fit.py`, accepts explicit observations, geometry and RGB.
The reporter adapter reuses round-27 witness 0 and round-29 candidate coordinates.
Camera focal/pixel aspect come from the reporter solve, not the artist's rig.
Camera geometry and the source annotations remain assumptions and assistant
judgments. This is not an automatic estimator for arbitrary footage.

Each eye has one correction shared across observations: projected globe x/y
translation bounded to one native pixel each, and yaw/pitch bounded to five
degrees each. The four parameters transfer unchanged to held observations.
They are local calibration corrections, not per-frame inverse tracking.

All arms retain the 53 broad spherical-Gaussian (SG) columns and add one SG.
Its initial direction is the central reflected ray of the brightest fitting
sample by mean RGB. Its two tangent offsets are bounded to 20 degrees per axis;
concentration ranges 128..8192, starting at 512. The finite-difference Jacobian
uses normalized central steps .001 and the solve reserves .002 inside the bounds.
The actual widest-lobe endpoint is therefore concentration 128.3554.

The four arms are fixed seed/fixed geometry, optimized light/fixed geometry,
optimized light/shared geometry, and the latter with spatial pupil backing.
All use a 17x17 source-lit iris grid. Backing is either constant or another
17x17 grid; the latter changes capacity and prior dimension. Neither is a
source-light-driven iris albedo or a physical pupil material estimate.

At BOX7, four sweeps alternate nonnegative coefficient fits and bounded local
least squares with at most 16 nonlinear evaluations per sweep. Appearance has
the inherited 1e-3 mean coefficient penalty; reflected radiance retains the
1e-6 full-sphere mean-square penalty, cross terms and 1e-6 coefficient coercivity.
Normalized geometry corrections have a 1e-4 squared penalty. All original
training RGB stays in the objective. Each invalid transport pixel contributes
an additional unit squared residual before sample normalization. A sweep is
accepted only if the complete objective does not increase.

This uses the bounded local solver documented by
[SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
The implementation's optical conventions retain
[PBRT's specular interface](https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission).
Neither source validates these chosen bounds, priors, initialization or real eyes.

## Reflection-covered evaluation

There are eight full fits and 32 whole-frame held fits. Folds hold source
730/760/790/820 individually. Every fold initializes its light and fits only
from its other three frames' RGB. All full and fold predictions for both eyes
are persisted before either seen 805/815 RGB is opened for scoring.

Support intersects every fitted arm's transport validity, two-frame appearance
occupancy and five-degree/all-node angular proximity, then intersects the frozen
round-29 evaluation mask. This makes some support adaptive to fitted geometry;
losses are explicit and these scores are not paired improvements over round 29.

| Held source | Left reflection / other | Right reflection / other |
|---|---:|---:|
| 730 | 22 / 68 | 35 / 136 |
| 760 | 2 / 130 | 27 / 145 |
| 790 | 2 / 130 | 28 / 165 |
| 820 | 6 / 83 | 22 / 173 |

At least eight reflections/eight other pixels make a fold eligible; selection
requires two eligible folds. Left has one and abstains. Right uses all four.
Compared with the frozen masks, left loses one pixel at 820 and one at 815;
right loses two at 730 and one at 805. Reflection counts are unchanged.
Seen 805 support is 41/203 left/right, with 0/35 labelled reflections.
Seen 815 support is 180/80, with 16/38 labelled reflections.

All table entries use identical support across round-30 arms. Display error
is mean RGB absolute sRGB error per native pixel, on a 255-level scale.

| Arm | Left fitting reflection median | Right fitting reflection median | Right balanced fold MSE | Left seen-815 reflection median | Right seen-815 reflection median |
|---|---:|---:|---:|---:|---:|
| Fixed seed | 53.029 | 48.747 | .0264951 | 63.909 | 58.429 |
| Light fit | 43.994 | 39.314 | .0271758 | 59.278 | 61.620 |
| Pose + light fit | 53.035 | 44.601 | .0278678 | 63.907 | 60.288 |
| Spatial backing | 31.709 | 24.857 | **.0150917** | 48.487 | **77.695** |

Right's selected spatial model has seen-815 reflection p95 **159.587** and
linear MSE **.1002623**, versus the fixed arm's .0703360. Left's smaller spatial
median again hides a worse linear MSE: .0468185 versus .0359714 fixed.
This is neither a reflection repair nor clean-layer uncertainty estimation.

On fitting reflection pixels, right coupled backing/reflection mean radiances
are .012664/.165573. Spatial backing instead assigns .231339/.012287.
Left spatial backing similarly assigns .235999 to backing and only .004457
to reflection. It largely paints the compact photographed spot below the cornea.
Decoded still comparisons confirm the wrong broad spot in the physical-light
arms and the spatial model's displaced spot at 815.

## Solver limitations and supplied-geometry controls

Compact bilinear accumulation avoids dense per-node material grids. Across eight
signed pose/translation cases it agrees with the old dense sampler to
**4.44e-16**. Independent scalar spherical optics agree on translated central
directions/Fresnel to **2.70e-11**, and same-node integrated RGB to **7.49e-11**.
These checks validate numerical conventions for supplied geometry, not anatomy.

The unknown-axis fixture uses independent scalar RGB for three translated
spherical eyes and a fourth held position. The initializer chooses a ray
**33.681 degrees** from the true light. Its bounded tangent search can reach
at most about **27.24 degrees** from that seed, so the true localized axis is
outside this search even before optimization. In every arm the added lobe's
RGB coefficients are exactly zero. With these held fixed in the nonlinear
step, moving the lobe cannot affect its contribution or radiance penalty.
This is an inactive-component failure as well as a poor initial neighborhood.

| Independent unknown-axis fixture | Held sum max linear error | Held layer max linear error |
|---|---:|---:|
| Fixed seed | .0246736 | .0478343 |
| Light fit | .0246736 | .0478343 |
| Pose + light | .0247279 | .0478977 |
| Spatial backing | .0475209 | .0540543 |

No positive automatic-localization claim is supported. The fixture deliberately
retains its failed initialization and wrong layers; no known axis was supplied
to the fitter and no target RGB was used to rescue it.

On the real fits, **66 of 120 nonlinear sweeps** exhaust their evaluation limit.
All accepted objectives descend; zero proposals are rejected. This does not
establish a local minimum of the full alternating problem. At most one original
training pixel has invalid transport in some coupled fits and remains in the
loss with its penalty. No hard geometry certification follows from that loss.
Full light-only fits on both eyes reach the broad concentration bound 128.3554.

Full right coupled correction is (-.0792, .1895) native pixels and
(.0263, .1183) degrees yaw/pitch. Full left coupled translation stays near
.0011 pixels, with pitch change -.0134 degrees. Extra parameters do not
guarantee a better attained solution: each arm starts from the fixed seed,
not from the optimized light-only solution. The coarse aperture classification
also makes the sampled objective piecewise smooth. We have not isolated that
effect as the cause of stagnation or measured derivative convergence.

Post-fit boundary checks are descriptive, not selection criteria. The inherited
left source-815 pupil discrepancy is 2.770 pixels; the coupled correction leaves
2.766 pixels. Right coupled source-815 pupil discrepancy is 1.576 pixels.
Reused source boundaries and small parameter updates do not validate 3D optics.

## Source sampling sensitivity

All full-fit coefficients are held fixed while evaluating BOX3/7/15/31.
The table gives the maximum RGB-channel sum difference from BOX31 across the
six observations, on each arm's common valid pixels at all four densities.
Support differs between arms, so this table is sensitivity, not paired ranking.

| Arm | Left BOX3 / BOX7 / BOX15 vs BOX31 | Right BOX3 / BOX7 / BOX15 vs BOX31 |
|---|---|---|
| Fixed seed | .0139325 / .0057624 / .0015848 | .0225951 / .0038717 / .0006771 |
| Light fit | .0112760 / .0047292 / .0013907 | .0118849 / .0020541 / .0004165 |
| Pose + light | .0144025 / .0057394 / .0022028 | .0202611 / .0034673 / .0008914 |
| Spatial backing | .0304188 / .0056083 / .0029771 | .0120758 / .0024871 / .0015595 |

Mixed iris/pupil footprints produce the largest left sum differences. Component
differences can exceed sum differences through cancellation: the spatial
backing component changes by up to .0093116 left/.0057253 right at BOX7 vs31.
Interior reflection sampling remains measurable on right. BOX31 is a denser
reference, not a convergence proof. The reporter's actual PSF, motion blur,
defocus and chroma sampling are still not inferred.

## Verification and review

All **40 final coefficient solves, saved forward predictions and support masks
replay exactly**. The entire nonlinear source-730 held fold is independently
rerun for each eye and replays its parameters, coefficients and history exactly.
The other 38 nonlinear trajectories are not rerun. Held RGB access is guarded;
805/815 fitting reads are zero. Frozen inputs and earlier evidence are unchanged.
**300 exact keyed-eye fallbacks, 304 protected hashes and 12 disabled complete
transactions pass.** No existing production numerical module was edited.

- `final/r30_continuous.mp4`: 150 continuous frames, 60000/1001 fps, 768x960,
  2.5025 seconds. Source left, sparse sampling overlays right. All seven decoded
  contact sheets inspected; both blink sequences retain the original eyes.
  Maximum frame mean encode error 1.686323 display levels.
- `final/r30_layers.mp4`: 144 frames at 24 fps, 1200x580, six seconds of labelled
  one-second still holds. Source, four model sums and coupled reflection share
  displayed native coordinates. All six decoded holds inspected. Black is
  unscored, and zero model pixels clip above white. Maximum frame mean encode
  error .407545 display levels.

The continuous review does not show a continuous reconstructed eye. These are
display-sRGB CRF12 reviews, not HDR masters, live playback or artist acceptance.
Encoding averages include black/unchanged pixels; they are not repair metrics.

Before any fresh final test, the solver needs a training-only initialization
that can reach candidate source directions and activate new light components,
plus optimization/derivative controls at material boundaries. Multiple starts
or coefficient refitting during light moves are candidates to test, not proven
solutions. Preserve explicit spatial-backing competition and failed source sums.
Source-driven iris illumination, credible 3D lid/light visibility, posterior
optics and actual scene-light registration/replacement remain required physical
work. Original catchlights, crown sheen and lip gloss are unresolved.
