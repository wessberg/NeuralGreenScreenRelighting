# Observed residuals still need correct reflection and geometry

Round 23 evaluates subtraction that retains the source's observed detail and
fits real optical proposals separately for each eye. The reporter's catchlights
remain in the exact keyed source. No complete eye reconstruction, inferred scene
reflection, production replacement or photorealism is claimed.

## What residual preservation does

For `source = clean + old`, a fitted decomposition gives `fit_clean + fit_old`.
`eye_residual.py` computes `source - fit_old`, equivalently `fit_clean` plus the
entire observed fitting residual. Its clean error is exactly `old - fit_old`.
This avoids discarding detail merely because the interior grid cannot represent
it. It also retains noise and any old reflection the model failed to explain.
If pupil radiance was assigned to reflection, subtraction still removes it.
Negative outputs remain raw diagnostic values; clamping is not a repair.

Analytic controls exercise all four cases: fine observed texture with oracle
reflection, a perfect sum that misassigns half the pupil radiance, source noise,
and over-subtraction yielding negative radiance. These controls do not certify
the learned reflection on an actual eye.

## Untouched synthetic test result

Final test has 1,440 common visible pixels: 909 pure iris, 149 pure pupil and
382 mixed-boundary pixels. Frame 17 is closed. All four fixed geometry arms
use 64/12/32 iris/pupil/environment grids and 7,009 training observations.
All twelve bounded channel solves pass the independently computed 1e-7
normalized-gradient gate (8.85e-9..9.97e-9). Selection source-sum MAE is
.812152 nominal, .982591 cornea, .856248 pupil and 3.415446 lid. These scores
are reported for sensitivity; no arm is selected as certified material.

Display RGB MAE, 0..255, on identical final-test masks:

| Region | Source | Fitted clean, nominal | Source minus reflection, nominal |
|---|---:|---:|---:|
| Whole visible eye | 5.06371 | .99598 | .71844 |
| Pure iris | 1.38765 | .51199 | .18167 |
| Pure pupil | 31.75991 | 2.82157 | 5.05011 |
| Mixed boundary | 3.39827 | 1.43560 | .30616 |

Residual preservation improves the iris and boundary means but worsens pupil
error relative to the fitted clean alternative. It produces **15 negative
pupil colour channels**. The negative values remain in linear scores and
diagnostics; the sRGB display transform floors negatives for viewing. The
review shows dark mottled pupil patches. Better whole-eye error does not satisfy
identity preservation or complete layer recovery.

Residual clean whole-eye / pupil MAEs for perturbed geometry are .90556/6.08848
(cornea +.3 mm), .71917/5.01192 (pupil +.2 mm), and .70730/4.95621 (lid -.15 mm).
The wrong lid even slightly improves these average residual scores. That does
not establish correct geometry: adding all unexplained source signal back can
hide a poor geometric fit. No true geometry was inferred by these sensitivity
arms, and no post-test tuning was performed.

Nominal residual plus oracle new reflection has whole-eye MAE .503421 against
target; add-only is 4.402395. The new term is the independent renderer's
`target-clean`, not a reflection inferred from the production scene. Its raw
minimum is -4.91738e-7. The same oracle term is used for every coupled arm.
Native screen-space temporal error-change MAE falls .00109809 to .00085846
linear across the full now-seen sequence; this is not material-aligned flicker,
an untouched temporal test or real-performance validation.

The fixed .002 green-channel training box is infeasible. Bounded minimax
linear programming gives a minimum uniform radius .0078334265. This radius
measures discrepancy conditional on this basis, geometry, quadrature and render
noise. It is neither camera noise nor a calibrated uncertainty estimate. Thus
this test cannot establish clean-material ambiguity at .002 precision. No
globally expanded confidence box is used to manufacture a material certificate.

Two LP runs (default HiGHS, then interior point with crossover) were interrupted
after the four image fits had completed. All fits were rerun with unchanged
data and objectives; the repeated solver outputs agree. `fit_r23_ipm.py` uses
interior point without crossover and records its settings. The completed LP
log includes an **imprecise interior-solve warning**, followed by final Optimal
status, reported dual residual 7.78e-9 and objective gap 8.27e-10. Verification
checks a feasible primal witness and constructs an independent feasible dual
lower bound rather than trusting that final status alone. No final-test RGB
was opened until the completed fit/selection/feasibility report was saved.

## New rendering and boundary correction

The [protocol](r23-eye-protocol.md) freezes 24 fresh 64x64 Cycles frames, with
16 training, four selection and four final-test frames. Texture phases, mark,
gaze phase and reflected environments differ from round 22. The actual BOX
width is one pixel, integrated at 8x8 for fitting. Independent 32x32 texture
lookup checks training frame 0. This is a fixture filter, not a measured camera
PSF. There is no lens motion blur or diffuse iris relighting: the interior is
emissive and the anatomy is an open anterior cap over a planar interior.

The pre-fit check exposed missing direct rays beside that finite cap. The
[correction](r23-transport-correction.md) adds their direct plane intersections
in an isolated fixture transport module. No old transport or production stage
was changed. The initial failed report is preserved. Corrected frame-0 clean
MAE is .307978 display levels at 8x8 and .045805 at 32x32; mixed-boundary MAE
is 1.063463 and .126662 respectively. Fitting retains the prescribed finite
quadrature and its error. Masks include direct, refracted and mixed pixels;
all arms use identical nominal geometry masks, including boundaries.

The frozen fit archives also retain the inherited coefficient-count support
flag. This is vacuously true for an empty matrix row. Because the common mask
includes footprint corners beyond the finite quadrature nodes, some visible
pixels have no modeled ray. The verifier therefore reports a separate corrected
`nonempty_three_frame_support` count, explicitly rejecting those rows; the raw
flag is never treated as complete observation coverage. No pixels are removed
from scores, and finite node coverage still cannot certify continuous support.

## Real optical proposals and why they are rejected

`audit_r23_geometry.py --shot` profiles 7.2/7.8/8.4 mm corneal curvature radii
at six evenly spaced stored frames for each eye. It fits image position, depth,
yaw, pitch and pupil radius against separate EllSeg pupil/limbus contours.
Three starting poses are tried. Camera intrinsics are the reporter shot's
explicitly assumed 50 mm/36 mm configuration; the anatomical iris radius,
conic shape, interior depth, cap extent and IOR remain assumptions.

Alternating contour samples are fit and checked, but both halves come from
the same predicted ellipses. These are **not withheld native RGB, independent
source annotations or anatomical truth**. Thirty-six profile fits have radial
check RMS .050970..454805 native pixels. No optimizer hits its parameter bounds.
Nevertheless all 36 fail complete cap support: pupil contour rays hit, but
limbus contour rays lie beyond the inherited 2.18 mm finite cap. The smooth
conic continuation used by the residual is not an actual finite-surface hit.
Every such profile is explicitly rejected. Their low contour error cannot
be reported as a solved physical cornea.

At the same proposed pupil-boundary pixels, the radius endpoints imply median
reflection-direction differences of 3.6312..6.2373 degrees across the 12 eyes.
Those are model sensitivity measurements, not measured true-ray errors. The
complete-ellipse versus RITnet visible-part IoU ranges .48718..83399. This
compares two proposals and cannot determine which is right. The review shows
small proposed pupil contours surrounding bright source catchlights, and
complete limbus contours continuing over lids in narrowed eyes. A 2D class
proposal neither certifies the pupil underneath a reflection nor supplies 3D
lid/eyelash occlusion for a new light. No 3D lid geometry was recovered.

## Published inverse-eye methods

The current primary pages distinguish two methods often called EyeNeRF.
[Seeing the World through Your Eyes (CVPR 2024)](https://world-from-eyes.github.io/)
jointly fits corneal pose, reflected scene and iris texture, with a texture
prior supporting reflected-scene reconstruction. It is the repository already
cloned under `dependencies/eyenerf`. Existing local source inspection records
cross-eye texture routing, rotational regularization and normalized preview
output; those must be addressed before claiming same-eye radiance preservation.
Residual subtraction alone cannot certify its learned reflected layer.

[EyeNeRF (SIGGRAPH 2022)](https://ligethz.github.io/eyenerf_website/) instead
combines explicit eye surfaces and implicit interior/periocular representations
for view synthesis and relighting from multiple cameras and lights. Its capture
demonstration does not establish recovery from this ordinary performance plate.
Neither full method was installed or trained this round. No new pretrained
model ran; stored EllSeg/RITnet proposals were reused. These limitations do not
establish impossibility and do not introduce a capture-reference requirement.

## Verification and review artifacts

The completed verifier independently samples all 96 saved frame/arm layer
predictions (maximum difference 1.665e-16), replays numerical gradients, support
and scores, and checks residual identities to 2.776e-17. Four guarded training
accessors, eight heldout-input substitutions and 52 poisoned real loader paths
check the data boundary. These establish input invariance, not repeated poisoned
end-to-end solves. A second LP solve supplies dual multipliers; after explicit
dual-feasibility correction, the lower bound is .00783339525 versus the primal
.00783342651, a 3.126e-8 gap. This supports infeasibility of the .002 model box.

Nominal nonempty three-frame coefficient support is 1,369/1,440 visible test
pixels. Its raw coefficient flag reports 1,397, including 28 empty modeled
rows. Corrected support is 1,381 cornea, 1,369 pupil and 1,322 lid; the lid
arm has 88 empty rows. All remain conditional finite-quadrature counts, not
complete material or continuous-footprint certification.

Independent 32x32 texture integration on open frames 0/5/11/23 gives clean MAE
.040785..045805 and source MAE .118264..160856 display levels. Closed frame 17
is exactly zero. Ninety-six disabled coupled transactions and twelve inherited
transaction controls pass. **All 300 real keyed-eye fallbacks and 304 protected
hashes pass.** The existing real fallback video retains its round-20 hash.

`final/r23_eyes.mp4` contains 72 frames, 24 fps, 960x762, three seconds: 24
chronological synthetic frames repeated three times, six labelled panels.
All decoded frames were checked against their PNGs; three ordered contact
sheets and decoded boards 5/23 were inspected. Maximum mean encoding difference
is .302531 display levels. Pupil patching survives encoding. This is not live
playback or artist acceptance.

`final/r23_geometry.mp4` contains 48 frames, 24 fps, 640x696, two seconds:
six chronological real snapshots held eight frames each, with separate eyes,
exact source on the left and proposal overlays on the right. It is a sampled
geometry review, not continuous performance motion. Both decoded contact sheets
and native board 0 were inspected. Maximum mean encoding difference is 1.551717
display levels. Source identity is verified on linear keyed files, not lossy
review pixels. Both videos are display-sRGB/CRF12 reviews, not HDR delivery.

Render and completed fit subprocess exit codes are zero. Frozen code snapshots,
72 EXRs, fitted coefficients, primal/dual witnesses and review images remain
under `composite/r23_eyes/`; all structured reports are under shot `logs/r23_*`.
No key, original footage, production compositor, prior round code or defaults
were changed. The latest generic review encoder log was updated by encoding
and is excluded from the round commit; existing unrelated work remains intact.
