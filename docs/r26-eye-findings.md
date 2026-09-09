# Shared eye geometry still leaves material correspondence ambiguous

Round 26 connects five source observations of each eye through shared optical
parameters and checks same-eye donor RGB against a withheld target. The best
boundary-conditioned model gives mixed observed-RGB results: the three-or-more
donor mean is slightly worse than a centre/scale control on the left and slightly
better on the right. Compatible geometry alternatives still disagree about
correspondence by several native pixels. **No clean iris, reflection removal,
scene-light replacement or photorealism is established.** Original keyed eyes
and catchlights remain unchanged.

## Source evidence and joint model

The source 730/760/790/820 observations from round 25 are reused development
evidence. Source 805 (local frame 105), separately per eye, adds source-only
assistant annotations made after the protocol was frozen and before fitting:
nine visible limbus points, four readable pupil points, an observed lid opening
and incomplete reflection exclusions. No hidden upper arc was invented and no
neural contour was loaded. Native RGB, origins, archive files, labels, camera,
prior fits and protocol hashes are recorded in `logs/r26_freeze.json`.

Source 805's first and last arc thirds fit its ring; its three middle points
are reserved. Only its first two pupil points condition geometry; the last two
are checks. Earlier pupil marks are development conditioning. This is a new
annotation/computational check on reused footage, not unseen-frame acceptance,
independent human truth or certified anatomy. The source-only boards were
visually inspected before annotation; the algorithm does not use target RGB
for fitting, correspondence selection, support or donor predictions.

`joint_eye_geometry.py` constrains all five observations of one eye with one
shared tuple: corneal radius 7.2/7.8/8.4mm, iris depth 3.3/3.6/3.9mm and pupil
decentration x/y from {-0.6,0,0.6}mm. Pose, image centre, conditional distance
and pupil radius vary per frame. Pupil radius is sampled from 1.2 to 3.0mm in
0.3mm steps. Round-25 training-only ring candidates are reused for the donors;
the new target uses the same signed orientation grid and train-only fitter.

Each shared tuple retains every compatible per-frame assignment. All five
observations must have at least one assignment under that tuple. Ranking uses
the summed squared training limbus and first-order pupil-boundary distances,
with no RGB or held boundary scores. These are joint shared-shape constraints,
not a temporal smoothness penalty or continuous gaze tracker. The table of
assignments and factored families is stored under `composite/r26_joint/`.

The numerical functions take source points, camera and optical inputs; the
fixed frames and assistant annotations in the audit are reporter diagnostics.
They do not define geometry or thresholds for other footage. The reporter's
shot camera and pixel aspect are used, with its assumed calibration and the
inherited 5.85mm scale prior. The artist's anamorphic camera is not substituted.

## Geometry and sensitivity

| Joint family diagnostic | Left eye | Right eye |
|---|---:|---:|
| Tested per-frame optical rows | 205,821 | 210,924 |
| Compatible shared tuples at 2px | 34 / 81 | 57 / 81 |
| Compatible per-frame assignments in those tuples | 16,278 | 46,605 |
| Shared tuples with centred pupil restriction | 1 | 9 |
| Shared tuples with iris depth fixed at 3.6mm | 11 | 18 |
| Shared tuples at 1px pupil tolerance | 4 | 26 |
| Shared tuples at 3px pupil tolerance | 71 | 81 |

The selected left tuple is (8.4mm, 3.9mm, -0.6mm, +0.6mm); the right tuple is
(7.2mm, 3.9mm, -0.6mm, 0mm). Several selected parameters sit at grid limits.
All ten frame/eye families still reach both -40 and +40 degrees yaw. Neither
the selected parameters nor a smaller count is a confidence estimate.

The new target's held native-pixel boundary distances are:

| Check | Left | Right |
|---|---|---|
| Joint-selected limbus, three points | .125941, .132415, .408211 | .865320, 1.157500, .024454 |
| Frontal limbus control, same points | .505054, .855249, .355277 | .715669, .959153, .156489 |
| Joint-selected pupil, two points | .751946, .436356 | .096953, .056153 |

All ten checks fall within the predeclared 2px band. That limited boundary
agreement does not identify hidden anatomy or the correct material mapping.
The best target gaze is (-30,0) degrees left and (0,-10) right, conditional on
the model and annotations; these are not measured gazes.

Pupil decentration, denser pupil-size sampling and depth freedom remove the
previous empty-family situation, but their effects are confounded. This is not
proof that pupil decentration was its cause. Shared families represent many
combinations, not independent samples: about 1.083 trillion left and 18.188
trillion right possible discrete sequences are represented without enumerating
their Cartesian products. Those counts are bookkeeping, not statistical evidence.

After the frozen primary scoring, exploratory controls shift all training pupil
marks together by one pixel, separately in each signed image direction, and
offset the pupil-size grid to 1.35..2.85mm. They do not refit or replace the
reported primary result. Their exact results are in `r26_verification.json`.
The four signed one-pixel shifts leave 12/49/41/24 shared tuples left and
40/71/72/44 right (x-/x+/y-/y+). The midpoint radius grid leaves 34 left and 59
right; its selected right corneal radius changes from 7.2 to 8.4mm. The same
corneal-radius change also occurs under the right eye's negative-x label shift.
They address only sampled label/grid sensitivity, not uncertainty in the camera,
limbus labels, smooth anatomy or the full optical model.

## Withheld observed-RGB transport

For the best conditioning-only joint hypothesis, native target pixels are
traced to the refracted iris plane and numerically mapped back into each donor
of the same eye. This assumes rigid iris coordinates and zero relative torsion.
All four bilinear donor pixels, each with 3x3 BOX quadrature, must be candidate
iris and remain conservatively within the source lid opening and outside its
reflection polygon. The margin is 2px plus sqrt(2) for the raster and pixel BOX.
Target pixels also pass candidate-iris BOX and source-opening tests.

The control uses frontal-ring centre/scale transport. Both methods are scored
on their identical paired support. Target RGB is opened for numerical scoring
only after coordinates, eligibility and donor predictions are saved to disk.
There is no RGB optimizer, appearance-based donor ranking, target-context fit
or target frame in the donor pool. The four donors are source 730/760/790/820.

Display RGB absolute error is averaged over channels at each native target
pixel, then summarized over those pixels; units are 0..255 display levels.
Donor means are formed with equal weights in linear RGB before display scoring.

| Eye / predictor | Paired target pixels | Joint median error | Centre/scale median error |
|---|---:|---:|---:|
| Left / source 730 | 92 | 7.005724 | 5.936690 |
| Left / source 760 | 142 | 6.171988 | 5.955445 |
| Left / source 790 | 125 | 4.753406 | 4.963929 |
| Left / source 820 | 134 | 6.881888 | 6.741157 |
| **Left / mean with 3+ donors** | **117** | **5.819331** | **5.561056** |
| Right / source 730 | 111 | 5.354574 | 5.308061 |
| Right / source 760 | 136 | 5.630217 | 5.267241 |
| Right / source 790 | 108 | 5.443773 | 6.777205 |
| Right / source 820 | 116 | 6.582845 | 7.741284 |
| **Right / mean with 3+ donors** | **110** | **5.412190** | **5.948307** |

The mean abstains on 254 of 371 eligible left pixels and 256 of 366 right
pixels. Eligibility here means conservatively within annotated open regions,
before candidate material and donor support. Pixel/donor rows are correlated.
The reported error is prediction of photographed radiance, not clean-material
accuracy. Unlabelled soft reflections, source illumination, blur and model
errors can all remain. No reflection-polygon target is being reconstructed.

Boundary-compatible geometry alone supplies direct counterexamples to unique
correspondence. Using just the best assignment at each retained shared tuple,
the median lower bound on donor-coordinate diameter is 3.165395px left and
2.955371px right. This is a subset
of the full factored family, and uses successful centre-ray mappings without
requiring every witness's full lid/BOX footprint. It describes geometric
ambiguity, not the number of usable donors. Added -10/+10 degree relative
texture-orientation witnesses test a further unmeasured assumption. Agreement
among these witnesses would still not certify the unenumerated family.

Changing lid/reflection uncertainty to 1px or 3px changes support substantially;
the fixed 2px result remains the primary check. No photometric threshold or
agreement count produces a reliable geometry, clean material or removal mask.

## Corrected solver failure and verification

The first inverse implementation initialized at the unrefracted iris projection.
Seven signed/asymmetric-camera controls exposed six failed inversions at oblique
gaze; four occurred at (30,-25) degrees and two at (-30,25). The largest returned
coordinate error was 3.035406px. They were flagged as invalid, but the control
correctly rejected the inability to recover reachable test rays.

Starting at the projected corneal apex and accepting only descending Newton
steps fixes all 2,513 rays in those controls. Maximum image-coordinate roundtrip
error is 2.232696e-9px. The centred-pupil residual differs from the previous
finite-difference implementation by at most .0000438192px. These are numerical
consistency controls, not independent anatomical or renderer validation.
Initial source predictions/report are preserved in `r26_joint/initial_inverse`;
the corrected audit was rerun, with effectively unchanged primary source scores.

The verifier replays every table row, factored family, selected mapping,
prediction, witness and score. Replacing held pupil labels or table check
columns leaves conditioning and selection unchanged; substituting entire held
limbus arcs preserves nine train-only profiles per eye. Two target-RGB read
guards per eye intercept a hypothetical zero/1000 replacement and confirm the
predictor never requests target RGB; predictions remain identical. Original
archives are never modified. Complete interpolation-footprint, closure, empty
family, signed decentration and independent bilinear-image controls pass.

The inherited 300 exact keyed-eye fallbacks and 304 protected hashes, plus ten
new disabled complete transactions, are checked. Geometry, correspondence,
clean-material and production-replacement flags remain false.

## Reviews and remaining work

- `final/r26_joint.mp4`: continuous 150 frames at shot fps, 768x960. Original
  source left, diagnostic right, eyes separate. Observed frames have annotations;
  other frames explicitly abstain. The new RGB check locations appear only at 805.
- `final/r26_detail.mp4`: five labelled one-second observation holds at 24fps,
  768x992. This makes the sparse annotations readable and is explicitly not motion.
- `final/r26_transport.mp4`: five labelled one-second target/donor comparisons
  at 24fps, 768x722. Source check, joint prediction and centre/scale use the same
  pixel support. Black means unscored, not black pupil truth or a repaired eye.

All three encodes and the full verifier exit zero. Seven decoded continuous
contact sheets, the decoded target board and all five distinct decoded holds
from each detail/transport review were visually inspected. Both source panels
retain the original catchlights through the continuous sequence, including its
blinks; overlays occur only at the five observed frames. Transport comparisons
show small lower/side iris regions with brightness/texture differences, while
the reflection areas remain outside the check. Maximum mean encode differences
are 1.892521, 1.494803 and .070712 display levels respectively. These image means
include unchanged/black areas and are encode checks, not material accuracy.
No live playback or artist approval is claimed.

These are display-sRGB/CRF12 reviews, not HDR masters or artist acceptance.
The continuous review preserves the original reflections. The held transport
review shows only small sampled parts of the observed iris; it cannot demonstrate
temporal material stability or reflection removal.

The sharp limbus and single effective refractive interface remain inherited
approximations. Varying iris depth does not model posterior cornea or lens.
Pupil deformation, credible 3D lids and reflected-light occlusion, smooth limbus,
camera PSF, material/source-light separation and real scene registration remain
unresolved. The coupled transaction is exercised only in its disabled source
fallback mode. No scene-light term is computed for this real fit.

The next step should let source photometry constrain a joint same-eye
material/geometry/light model with held target RGB and competing solutions.
It must account for moving reflections as a separate layer; copying the best
boundary fit's observed donor radiance cannot establish clean iris texture.
Complete source removal plus scene-derived replacement remains the task.
