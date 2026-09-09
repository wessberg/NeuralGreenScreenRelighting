# Round 23 handoff — observed iris detail improves; pupil and geometry still fail

**Date:** 2026-09-09. Starts from round 22. Production defaults unchanged.
**Delivered:** residual-preserving subtraction diagnostic on fresh Cycles data,
finite-cap transport correction, fixed-tolerance model feasibility with primal
and dual evidence, separate real-eye optical proposal profiles, exact-source
verification and two labelled review videos.
**Not delivered:** real catchlight removal, reliable real corneal/pupil/lid
geometry, complete iris/pupil material, inferred scene reflection or photorealism.

## 1. Artist steering and scope

Stay on eyes; instrument task remains deferred. Preserve observed same-eye
identity, separate eyes, generic shot inputs, and the exact keyed source including
original catchlights unless the entire removal/replacement transaction has earned
support. No key retuning, generated performer texture or capture references.

Read `docs/r23-eye-findings.md`, `docs/r23-eye-protocol.md` and the pre-fit
`docs/r23-transport-correction.md`. Full neural inverse-eye methods remain unrun;
stored EllSeg/RITnet proposals were reused, with no new model inference.

## 2. Fresh residual-preservation experiment

`eye_residual.py` computes source minus fitted old reflection, retaining the
entire source residual. Algebraically its error against true clean is true old
reflection minus fitted old reflection. It preserves detail omitted by a coarse
clean grid, but also preserves noise and unresolved reflection, and still removes
pupil radiance wrongly assigned to reflection. Analytic controls verify all these
cases; a perfect source sum can still remove half the nonblack pupil radiance.

`render_r23_eyes.py` renders 24 fresh 64x64 nonperformer frames, as source/clean/
target: 72 EXRs, OptiX, 4096 samples, seed 23, actual BOX width 1, no persistent
data or denoising. New phases/mark/environments differ from round 22. Sixteen
training frames, four selection frames 2/8/14/20, four final test 5/11/17/23.
No round-22 test RGB was used. Frame 17 is closed.

`audit_r23_eyes.py` fits fixed 64/12/32 iris/pupil/environment grids to 7,009
training pixels. Four arms: nominal supplied geometry, cornea radius +.3 mm,
pupil radius +.2 mm, and lid half-height -.15 mm. These are sensitivity arms,
not real geometry fitting or newly selected anatomy. All twelve channel solves
pass the 1e-7 normalized-gradient gate (8.85e-9..9.97e-9).

The native pixel operator averages 8x8 rays; masks add footprint corners and
include pure iris, pure pupil and mixed boundaries. The BOX is a fixture filter,
not the reporter's measured PSF. Interior emitters, a single anterior interface
and planar lids remain simplified; diffuse iris relighting is not tested.

## 3. Missing transport found before fitting

The training-frame-0 32x32 direct texture check initially had clean boundary MAE
10.74118 display levels. The finite corneal cap is smaller than the iris plane;
rays that miss the cap can see that emitter directly, but inherited trace dropped
them. `finite_cornea_transport.py` adds those direct plane intersections with zero
Fresnel reflection and camera-aperture visibility. This is an isolated fixture
correction, not a realistic anatomical sclera or a change to old round code.

The corrected common mask has 483 frame-0 pixels instead of 457, so whole-mask
before/after means are not paired-pixel scores. Corrected clean MAE is .307978
at 8x8 and .045805 at 32x32; mixed-boundary errors are
1.063463 and .126662 respectively. No quadrature/texture/split/solver capacity
was retuned after the test. The initial failure is preserved in
`r23_transport_missing_direct.json`; the separate correction document was hashed
before fitting. No final-test or non-frame-0 reference RGB was open then.

## 4. Untouched test and model feasibility

Display RGB MAE, 0..255, same nominal masks for every arm:

| Region | Pixels | Source | Nominal fitted clean | Nominal source minus fitted reflection |
|---|---:|---:|---:|---:|
| Whole visible | 1,440 | 5.06371 | .99598 | .71844 |
| Pure iris | 909 | 1.38765 | .51199 | .18167 |
| Pure pupil | 149 | 31.75991 | 2.82157 | 5.05011 |
| Mixed boundary | 382 | 3.39827 | 1.43560 | .30616 |

Residual preservation improves iris/boundary means but worsens the pupil versus
fitted clean. **Fifteen pupil colour channels go negative.** Raw linear values
and errors are preserved; the display transform floors negatives for viewing.
The review shows dark pupil patches. This is not faithful complete material.

Perturbed geometry residual whole-eye/pupil errors: cornea .90556/6.08848,
pupil .71917/5.01192, lid .70730/4.95621. The wrong lid slightly improves mean
residual clean error while its source fit is much worse: adding unexplained
source signal back can hide geometry errors. Selection source MAEs are nominal
.812152, cornea .982591, pupil .856248, lid 3.415446. No arm earns certification.

Nominal coupled residual plus oracle new reflection has whole-eye MAE .503421;
add-only is 4.402395. New reflection remains oracle target-minus-clean, with raw
minimum -4.91738e-7, not inferred production-scene light. Screen-space error-change
MAE over all now-seen chronological frames is .00109809 fitted versus .00085846
residual. That is not tracked material flicker or real-performance validation.

The fixed .002 green-channel training residual box is infeasible. Minimax radius
is .0078334265116. An independent feasible dual lower bound is .0078333952532;
primal/dual gap 3.126e-8 and primal violation zero. The model is incompatible
with .002 tolerance, conditional on its geometry/basis/quadrature/render noise.
Do not infer ambiguity at camera-noise precision or universal impossibility.

Default HiGHS and then IPM with crossover were interrupted after the four image
fits completed. All image fits were rerun unchanged. `fit_r23_ipm.py` completes
with crossover disabled. The final LP log reports an imprecise interior solve
before postsolve changes status to Optimal; reported dual residual 7.78e-9 and
objective gap 8.27e-10. Our independent primal/dual checks support the tolerance
conclusion rather than trusting that status. SciPy warns that run_crossover is
forwarded to HiGHS; the log confirms crossover was not run. Fit and render process
exit codes are zero. Final-test RGB stayed closed until the completed report.

The frozen saved support flag is coefficient-count support and is vacuously true
for empty matrix rows. The verifier's corrected nonempty support is nominal
1,369/1,440, cornea 1,381, pupil 1,369, lid 1,322. Nominal raw 1,397 includes
28 empty rows; lid has 88. Keep those pixels in scores, reject empty rows as
observation support, and never treat finite nodes as continuous coverage.

## 5. Real geometry, verification and review

`audit_r23_geometry.py` independently profiles 7.2/7.8/8.4 mm corneal radii for
six stored frames per eye (source 700/730/760/789/819/849), fitting image position,
depth, yaw, pitch and pupil radius from EllSeg contours, with three pose starts.
Shot camera intrinsics and fixed anatomical dimensions are explicitly assumed.
Alternating fit/check samples belong to the same predicted ellipses: they are
not independent source annotations, hidden RGB or anatomical truth.

All 36 profiles have radial check RMS .050970..454805 native pixels, with no
parameter-bound hits, but **all fail actual finite-cap support**. Pupil rays hit;
limbus rays do not. The smooth conic continuation minimized by the residual is
not a surface hit on the inherited cap. Profiles are explicitly rejected.
Endpoint radii imply median reflected-ray differences 3.6312..6.2373 degrees
at the same proposed pupil pixels. This is sensitivity, not ray accuracy.
Complete EllSeg / visible RITnet IoU is .48718..83399, comparing two proposals.
The video shows small pupil contours enclosing bright catchlights and complete
ellipses crossing lids. No true pupil beneath a reflection or 3D lid was inferred.

Primary method pages were rechecked. CVPR-2024 Seeing the World through Your Eyes
is the locally cloned reflected-scene method, distinct from SIGGRAPH-2022 EyeNeRF
for eye-region relighting. Local CVPR code still routes both eyes through its left
texture network and uses rotational priors/normalized previews. Neither full
method was installed/trained; those outputs would not automatically certify
same-eye radiance. See the linked primary sources in the findings document.

Verification passes:

- 96 independent frame/arm direct grid samples, max 1.665e-16; residual identities
  to 2.776e-17; gradients, scores, coefficient and nonempty coverage recomputed.
- Four training access guards, eight heldout-input substitutions, 52 poisoned
  actual loader paths. Input invariance, not full poisoned end-to-end solves.
- Primal/dual LP evidence, analytic full-rank/ambiguous/unseen-coefficient controls,
  36 real profile replays, 96 disabled coupled transactions and 12 inherited controls.
- Independent 32x32 texture transport on open 0/5/11/23: clean .040785..045805,
  source .118264..160856 display MAE. Closed 17 is exactly zero.
- **300 exact keyed-eye fallbacks and 304 protected hashes unchanged.**

`final/r23_eyes.mp4`: 72 frames, 24 fps, 960x762, 3 seconds; chronological 24-frame
fixture repeated three times. Six panels show source, clean truth, fitted clean,
residual clean, target truth and residual plus oracle reflection. All three
decoded contact sheets and native boards 5/23 inspected. Maximum mean encoding
difference .302531 display levels; pupil patching remains.

`final/r23_geometry.mp4`: 48 frames, 24 fps, 640x696, 2 seconds; six chronological
real snapshots held eight frames each. Source left, proposed contours right,
separate eyes. This is not continuous performance motion. Both decoded sheets
and native board 0 inspected; maximum mean encoding difference 1.551717. Lossy
review pixels are not the exact-keyed-source identity check.

`final/r20_source_fallback.mp4` is reused by reference with its round-20 hash
unchanged. Original real catchlights remain. No live playback or artist acceptance.
All reviews are display-sRGB/CRF12, not HDR delivery. Frozen code, fits/witnesses,
72 EXRs and review images are in `composite/r23_eyes`; reports in shot `logs/r23_*`.
Existing AGENTS/round-17 work is preserved; the encoder's generic latest review
log is excluded from the round commit.

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' -b --factory-startup --python scripts/render_r23_eyes.py -- --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r23_eyes.py --shot music_videos/reporter_test/shots/reporter_a --transport-only
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/fit_r23_ipm.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r23_eyes.py --shot music_videos/reporter_test/shots/reporter_a --evaluate-only
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r23_geometry.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r23_eyes/review --fps 24 --out r23_eyes.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r23_geometry/review --fps 24 --out r23_geometry.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r23_eyes.py --shot music_videos/reporter_test/shots/reporter_a
```

## 6. Starter prompt for round 24

> Stay on eyes. Read round23.md and docs/r23-eye-findings.md. Keeping the observed
> residual improves synthetic iris detail but worsens pupil error and creates
> negative pupil radiance; subtraction does not solve layer attribution. The
> fixed .002 model box is demonstrably infeasible, so do not relabel its mismatch
> as uncertainty at measured camera noise. Real EllSeg contour fits are not eye
> geometry: all 36 profiles extrapolate beyond the finite cap at the limbus,
> and proposed pupils can enclose source catchlights. Address an anatomically
> coherent corneal/limbus/pupil/lid model and independent source-observation
> evidence for its boundaries and layers. Do not spend another round merely
> optimizing the same predicted ellipses or increasing solver iterations. If
> pursuing the full inverse-eye code, handle separate same-eye texture routing,
> priors, nonblack pupil radiance and actual scene-reflection coupling explicitly;
> the published neural methods remain unrun. Keep original keyed eyes exactly,
> including catchlights, until complete material, geometry and replacement
> evidence are earned. Never use empty basis rows as supported pixels. Preserve
> native filter/boundary modeling; round23 final test is now seen and must not be
> tuned into a fresh acceptance claim. No generated performer texture, key
> retuning, capture references, tracking-only detour or impossibility claims.
> Produce a labelled temporal review and round24.md.
