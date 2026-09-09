# Radiance is stable, but the pupil-covered reflection fit fails

Round 28 fixes the sampled-tail instability in the experimental objective and
includes actual labelled catchlight pixels through an explicit pupil/backing
path. **Their predicted appearance fails. No reflection removal or scene-light
replacement is earned.** Source eyes remain exact. The new numerical module is
input-driven; the reused reporter witnesses and annotations are a diagnostic,
not an automatic generic estimator.

## What changed

`eye_radiance_fit.py` fits nonnegative coefficients in the existing linear
working exposure units. Its objective per RGB channel is:

```text
mean_pixels((X c - observed)^2)
+ (1e-3 / appearance_count) * sum(appearance_coefficients^2)
+ lambda * (mean_sphere(L(direction)^2) + 1e-6 * sum(light_coefficients^2))
```

The three fixed reflected penalties are 1e-6, 1e-4 and 1e-2. The appearance-only
control omits reflected light. All four are tested for six reused geometry
witnesses per eye. Numerical column preconditioning transforms the entire
augmented system, including its penalty. Tiny training columns no longer make
enormous radiance cheap simply because the observations only touch their tails.

The 53 light basis functions are the existing constant and 52 spherical
Gaussians. For `b_i(w) = exp(v_i dot w - k_i)`, the analytic sphere-mean Gram
entry is `exp(-k_i-k_j) * sinh(|v_i+v_j|) / |v_i+v_j|`, with its limit at zero.
Cross terms are included. The small coefficient term makes the coefficient
penalty positive definite. These are declared exposure-unit priors, **not
calibrated SI radiance, measured source brightness, or a proof of correct light**.

`trace_connected` now accepts an optional decentered `pupil_offset`. It tests
the refracted ray against that aperture, retains the centered outer iris bound,
and requires the pupil ray to reach the recessed backing before the globe wall.
The default centered case reproduces every round-27 design and prediction.

The new forward design BOX-integrates three contributions: source-lit iris
appearance, an unknown constant backing RGB shared across fitting frames, and
reflected light. A native pixel may mix iris and backing samples. All nine BOX
samples must reach one of those surfaces. The backing is not assumed black;
neither it nor the iris grid is certified material. The single effective
interface retains the existing Fresnel/reflective optics, described by
[PBRT](https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission).
The constant backing and stationary appearance are experimental approximations.

Reflected angular sampling requires **every BOX node** to be within five degrees
of observed directions in at least two distinct fitting frames. This is a fixed
necessary sampling criterion for this experiment, not sufficient radiance
observability or a statistically validated acceptance threshold. Appearance
coefficients still require occupancy in two frames. Candidate, appearance and
angular support are intersected across all six witnesses for paired scoring.

## Controlled evidence

The protocol was frozen before fitting. Independent scalar fixture RGB uses
known appearance patches and analytic directional light, with Fresnel varied
separately to provide layer excitation. This is a numerical identifiability
fixture, not independently rendered anatomy or generated performer texture.

| Control | Measured result |
|---|---:|
| Analytic sphere Gram vs independent quadrature, max error | 2.11e-15 |
| Excited fixture withheld sum / layer max linear error | .000266171 / .000277567 |
| Duplicate-observation coefficient difference | 1.67e-15 |
| Sevenfold exposure-scaling coefficient difference | 5.33e-15 |
| Tiny-tail adversary, old / regularized maximum prediction | 2,181,619.86365 / .119999568 |
| Known withheld bright lobe, max linear prediction error | .478919483; angular gate rejects |
| Angular positive / unseen-direction rejection | 10/10 / 10/10 |
| One-frame-only / one-bad-BOX-node rejection | 10/10 / 10/10 |
| Four signed pupil shifts | 32 changed classified pixels each |
| Closed-eye sample rejection | 1,296/1,296 |
| Scalar pupil/iris BOX integration max error | 2.22e-16, including a mixed pixel |

The initial BOX fixture points covered pure iris/backing only. Inspection caught
that gap; the final verifier includes a fractional boundary-crossing pixel and
asserts a genuinely mixed sample count. This changed test coverage, not source
fitting parameters or thresholds.

A negative fixture has exactly identical observed radiance for `A=.08, L=1`
and `A=.1, L=.02`, both with `F=.02`. Their transmitted contributions differ
by **.0196 linear**, despite zero total error and full angular sampling coverage.
The gate does not resolve this ambiguity. Regularization picks a prior-favored
explanation; that choice does not earn clean material.

A separate same-design replay uses the original round-27 matrices with the new
objective. Across all 24 models per eye, maximum source-815 predictions are
.160059 left and .251808 right. The frozen old right family reached
201,740.159504. This diagnoses numerical stability on identical support; it does
not measure a reflection repair. Both appearance and light penalty definitions
changed, so it is not an isolated light-penalty ablation.

## Seen-source reflection evidence

Fit source 730/760/790/820, select on seen 805, and diagnose seen 815. No new
source image, annotation or final-test frame was consumed. Both eyes' predictions
are persisted before either 815 RGB array is read by the scoring stage. Source
815 remains **seen**, not fresh merely because numerical reads are separated.

| Support | Left | Right |
|---|---:|---:|
| Four-frame fitting pixels | 1,053 | 1,048 |
| Labelled reflection fitting pixels | 122 | 139 |
| 805 candidate / appearance / angular / joint pixels | 276 / 251 / 45 / 45 | 304 / 295 / 216 / 214 |
| 805 labelled reflection candidate / joint pixels | 30 / **0** | 40 / 38 |
| 815 candidate / appearance / angular / joint pixels | 319 / 268 / 207 / 207 | 314 / 279 / 131 / 115 |
| 815 labelled reflection candidate / appearance / joint | 37 / 36 / 16 | 41 / 41 / 41 |

Round 27's annulus-only fit contained eight/two labelled reflection pixels and
scored none at 815. Adding backing materially changes coverage. It does **not**
make new whole-support RGB scores comparable to the old annulus-only scores.
The incomplete polygons remain assistant source-only judgments on reused stock
footage, not independent human truth or certified clean material.

| Development-selected model | Left | Right |
|---|---:|---:|
| Index / geometry witness | 0 / 0 | 1 / 0 |
| Reflected penalty | none | 1e-6 |
| Fit linear RGB MSE | .007884762 | .008336739 |
| 805 linear RGB MSE | .000148006 | .009302121 |
| 815 linear RGB MSE | .006464097 | .025466827 |
| 815 median / p95 display error, all supported | 10.3661 / 90.6366 | 23.1435 / 110.2382 |
| 815 median / p95 display error, labelled reflections | **69.1028 / 93.0343** | **52.2301 / 123.5674** |
| 815 supported predicted pixels clipped above white | 0 | 0 |

Display errors are per-pixel RGB-mean absolute sRGB differences on a 255-level
scale, without upper clamping for scoring. Reviews clamp for display. The
left development gate excludes every labelled catchlight, so its low development
error cannot select for catchlight quality. The right reflected model beats its
appearance-only control on development MSE (.00930212 vs .01067911), but loses
on seen-815 MSE (.02546683 vs .02369824). This does not justify changing the
frozen selection rule using 815.

The failure is already present in fitting observations: the selected left
model's four labelled-reflection median display errors range 38.0372–65.7335;
the selected right model's range 48.6414–62.4116. Thus this is no longer just
unseen-angle extrapolation. The model cannot reproduce the training catchlights.
The diagnostics do not isolate whether light basis, geometry, PSF, visibility,
backing model or several together dominate that failure.

In the held review, the left model fits a grey pupil patch with backing RGB
(.175902, .151071, .142655); the right selected model fits a dark backing
(.012647, .013084, .020416) under a broad reflection. The right appearance-only
control instead fits a bright flat backing. These visibly incompatible and
inaccurate explanations are retained, not called credible uncertainty bounds.

## Verification, review and limits

All 48 new fits/designs/selections/predictions replay exactly. All 48 regularized
fits on old support also replay, and all 48 frozen round-27 fits and controls
remain exact. Two 815-RGB substitution/read-guard runs per eye produce zero
target reads in fitting and identical predictions. Twelve disabled complete
removal/replacement transactions preserve source exactly. All **300 keyed-eye
fallbacks and 304 protected hashes** pass.

`final/r28_radiance.mp4` is 150 continuous frames at 60000/1001 fps, 768x960,
2.5025 seconds: source left, sampling overlays right, sparse diagnostic status,
unchanged original eyes through motion and blinks. Seven decoded contact sheets
and decoded 815 inspected. Maximum frame mean encode error is 1.712402 levels.

`final/r28_layers.mp4` is 72 frames at 24 fps, 960x636, three seconds of explicitly
labelled one-second holds: selected model, appearance control, best reflected
model. All three decoded holds inspected. Source/sum/appearance/reflection share
identical scored support. Black is unscored. Maximum mean encode error is
.235328 levels. Reviews are display-sRGB CRF12, not masters, live playback or
artist acceptance. Their encoding error is not reconstruction accuracy.

The full task remains coupled source material/geometry/light inference and
actual scene-light replacement. This round deliberately completed prerequisites
without a fresh test because real catchlight prediction already fails on fitting
observations. Source-lit iris appearance is not albedo and is not illuminated by
the fitted environment. The oversized inherited spherical lid shell and 2D inset
mask do not validate 3D light visibility. Geometry remains a small fixed witness
subset, camera/scale assumed, with no continuous photometric pose fit, credible
posterior optics, torsion, physical iris-light coupling or actual scene mapping.
No neural model or Cycles render ran. Production defaults remain unchanged.
