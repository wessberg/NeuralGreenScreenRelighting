# Sharper pupil appearance does not establish reflected light

Round 29 crosses six reused optical geometry witnesses with 3x3 and 7x7 native
pixel integration, broader/narrower illumination and competing pupil backing.
**No model earns source removal or scene replacement.** The right-eye model
chosen by reflection-covered frame holdouts assigns the compact highlight mainly
to spatial backing, then fails seen source 815 worse than the broad-light arm.
The left eye abstains from selection because only one fold has enough supported
catchlight pixels. Production eyes, original catchlights and defaults are exact.

## Model and evaluation

The frozen protocol is `docs/r29-eye-protocol.md`. `eye_shape_fit.py` generalizes
the exposure-unit sphere penalty to supplied spherical-Gaussian vectors. The
baseline has 53 columns; the narrow model adds 48 lobes: concentrations 128 and
512 at 24 fitting-direction centres. Farthest-point centre selection uses
central reflected rays from fitting observations, without RGB or reflection
labels. Each held fold builds its own dictionary using only its three donors.
This is a limited deterministic dictionary, not optimized source light shape.

All arms have a 17x17 source-lit iris grid. Appearance-only, broad and narrow
arms have a constant backing; the competing arm uses a 17x17 backing grid at the
recessed plane. The fitted backing can retain the photographed highlight. Its
lower residual does not establish a plausible pupil interior. Nonnegative fits
retain the 1e-3 mean squared appearance coefficient penalty and the 1e-6
full-sphere mean squared light penalty, including cross terms and 1e-6
coefficient coercivity. The larger backing changes the appearance prior's
dimension as well as capacity; that comparison is not an isolated physical
parameter experiment.

The six boundary-derived geometries are unchanged. The new real forward designs
also require backing grid reachability and all 7x7 nodes, slightly changing
candidate support. Scores must not be presented as a paired improvement over
round 28. Camera focal lengths/pixel aspect come from the existing reporter
solve; camera and scale remain assumed. BOX integration describes a numerical
pixel model, not the reporter's measured PSF, chroma reconstruction, defocus or
motion blur. Reflection/refraction conventions are documented in
[PBRT](https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission),
and sampling rationale in [PBRT's reconstruction chapter](https://www.pbr-book.org/4ed/Sampling_and_Reconstruction/Image_Reconstruction).
Neither source validates these experimental choices or the real geometry.

Four leave-one-frame-out folds use sources 730/760/790/820. Candidate support,
two-frame appearance occupancy and five-degree/all-node reflected sampling are
intersected across all geometry/sampling/backing arms. A selection fold needs
at least eight labelled reflections and eight other pixels; selection needs at
least two eligible folds. Eligible folds equally weight reflection and other
linear MSE, then receive equal weight in the final score. Ineligible folds are
reported, not silently included as reflection evidence.

| Held source | Left supported reflection / other | Right supported reflection / other |
|---|---:|---:|
| 730 | 22 / 68 | 35 / 138 |
| 760 | 2 / 130 | 27 / 145 |
| 790 | 2 / 130 | 28 / 165 |
| 820 | 6 / 84 | 22 / 173 |

Thus left selection abstains; right uses all four folds. This was measured
before any new fitting. No threshold was relaxed. All these frames and labels
are reused, seen development evidence, not an independent final test. Both
eyes' predictions are saved before either 805/815 RGB is read for scoring.

## Independent controls

Fixture truth uses a separately implemented scalar sphere intersection,
Snell/Fresnel calculation, iris/pupil classification and pixel quadrature.
It does not call production ray functions. Its spherical eye and constant
iris/backing are intentionally limited geometry, not performer reconstruction.
The fixture's narrow dictionary receives the known true light axis; it tests
representation/transport, not automatic light localization.

| Control | Result |
|---|---:|
| Five signed positional cases, scalar vs production transport max error | 2.23e-11 |
| Full-sphere Gram vs independent quadrature | 4.85e-15 |
| Same-node scalar vs production integrated RGB max error | 2.27e-11 |
| 3x3 / 7x7 / 21x21 vs scalar 41x41 max linear error | .00416711 / .000117108 / .000117103 |
| Broad / narrow held sum max linear error | .0197942 / .00100541 |
| Broad / narrow held layer max linear error | .0260851 / .0223170 |

The narrow light reproduces the held sum more accurately but still gets layers
wrong. An explicit single-view spatial-radiance explanation has exactly the
same sum and zero reflection, with layer discrepancy .0279707. It is a
single-view counterexample, not a consistent temporal texture solution.
Its initial test expected discrepancy >.05; the actual held reflection peaks
at .02797, so the assertion was corrected to >.01. Geometry, light, sample
points and real protocol were not changed to improve this test. These are
diagnostic sanity thresholds, not acceptance gates for performer material.

## Real fitting and later seen failure

Across all twelve geometry/sampling combinations, narrow-light fitting
catchlight median display errors remain **43.776-52.487 left and 47.966-53.428
right**. Spatial backing lowers those ranges to 26.231-32.445 and
20.127-26.240, respectively; it does not make fitting exact or physical.

The following comparison fixes geometry witness 0 and BOX3 for every arm.
This is the right selected configuration; left is explicitly a fixed,
unselected diagnostic. All arms score identical native pixels.

| Arm | Left fit reflection median | Right fit reflection median | Right balanced fold MSE | Left seen-815 reflection median | Right seen-815 reflection median |
|---|---:|---:|---:|---:|---:|
| Appearance only | 51.169 | 61.882 | .0334558 | 69.102 | 63.341 |
| Broad light | 55.471 | 55.528 | .0287482 | 65.035 | 57.908 |
| Narrow light | 50.677 | 48.883 | .0255824 | 69.042 | 62.875 |
| Narrow light + spatial backing | 31.914 | 25.547 | **.0152352** | 47.827 | **74.557** |

Display errors are native per-pixel mean RGB absolute sRGB error on a 255-level
scale. Seen 815 joint support is 181/80 pixels left/right, including 16/38
labelled reflections. Its right selected reflection p95 is 157.444 levels.
Seen 805 joint support is 41/204, including zero/35 labelled reflections.
The left median reduction at 815 is not a pass: its spatial-backing p95 is
131.056 and linear MSE .0489275, versus broad .0301177.

The competing layers are numerically explicit. On right source-730 labelled
reflections, narrow constant backing contributes mean .003079 and reflected
light .125470 linear RGB. The selected spatial-backing arm contributes .200181
from backing and only .009489 from reflection. Source 805 is similar:
.251381 backing versus .010735 reflected. This assigns the photographed spot
primarily below the cornea; it is not evidence of recovering material beneath
the source reflection. All competing estimates are retained as failed fits,
not credible uncertainty bounds.

Sampling matters separately. Applying each BOX3 model's unchanged coefficients
at BOX7 changes candidate sums by up to .051818 left/.031220 right over all
arms and six observations. Restricting to narrow light gives .038787/.008622.
Refitting both densities changes sums by up to .063780/.031689 over all arms.
The largest differences occur in the appearance-only arms, so they cannot be
attributed solely to narrow reflections. These measurements do not establish
BOX7 convergence on the real eye or identify the unknown camera PSF.

All full-model candidate predictions remain below .711023 left/.801959 right;
no displayed predictions clip above white. This sampled stability is not a
bound on every unobserved direction. The spherical radiance penalty remains
necessary; angular proximity remains insufficient for layer observability.

## Verification, review and remaining physical work

All 96 full fits and predictions and 384 held-fold scores replay exactly.
Fitter access guards exclude held-fold RGB; seen 805/815 reads during fitting
are zero. Protocol/source/round-27/round-28 artifact hashes are unchanged.
All 300 keyed-eye fallbacks, 304 protected hashes and 12 disabled complete
transactions pass. No old production numerical module was modified.

`final/r29_shape.mp4` is 150 continuous source-left frames, 60000/1001 fps,
768x960, 2.5025 seconds. Seven decoded contact sheets were inspected; both
blink sequences preserve original source eyes. Sparse overlay colours denote
sampling and labels, not repairs. Maximum frame mean encode error is 1.681764
display levels. `final/r29_layers.mp4` has six labelled one-second still holds,
144 frames at 24 fps, 1000x580. All six decoded holds were inspected, including
source 815's displaced/incorrect predicted spot. Maximum frame mean encode
error is .415507. These means include black/unchanged pixels, not reconstruction
scores. Reviews are display-sRGB CRF12, not HDR masters, playback approval or
artist acceptance.

This round addresses source shape, discrete geometry witnesses and quadrature;
it does not perform continuous joint optical fitting. Credible 3D lid/light
visibility, source-light-driven iris albedo/transport, posterior eye optics and
actual scene-light registration remain open. No Cycles render or neural model
ran. A future physical solve must challenge the spatial-backing explanation,
resolve source-sum failure and measure sampling convergence before using a
fresh final check. A finer dictionary alone is not established as the answer.
Original catchlights, crown sheen and lip gloss remain unresolved.
