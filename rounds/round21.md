# Round 21 handoff — independent textures reject the current eye fit

**Date:** 2026-09-09. Starts from round 20. Production defaults unchanged.
**Delivered:** independent textured Cycles eye challenge, unknown pupil layer,
finite geometry sensitivity/selection, a corrected pixel-filter bug, numerical
and source-preservation checks, and a labelled temporal review.
**Not delivered:** a real reporter repair, reliable real eye geometry, complete
material support, performer scene-reflection replacement, or photorealism.

## 1. Artist steering and scope

Stay on eyes until fixed. The instrument task remains deferred. Preserve observed
performer identity, separate eyes, generic input support and exact keyed source
including catchlights whenever the full removal/replacement transaction is not
supported. Do not retune the key or require special capture references.

Read `docs/r21-eye-findings.md` and `docs/r21-eye-protocol.md`. The full EyeNeRF
method remains uninstalled/untrained; no pretrained neural method ran this round.

## 2. What was built and frozen

`render_r21_eyes.py` renders 24 nonperformer 64x64 frames, each as source, clean
interior and new-environment target: **72 EXRs**. The renderer uses 512x512 textured
iris, nonblack textured pupil and sharp environment patterns, independent of the
fitter's basis. Shared simplified anatomy is still one anterior conic/interface,
emissive planar interior, fixed pupil, planar lids and no scattering or eyelashes.
This is independent rendering/basis evidence, not realistic anatomy or real footage.

`eye_layer_fit.py` adds a separately fitted pupil grid with pupil-local coordinates,
without imposing black radiance or a texture prior. The isolated `audit_r21_eyes.py`
compares a continuous 12x12 interior with split-pupil fits under nine supplied
geometry hypotheses: true, yaw +/-2 degrees, radius +/-5%, pupil +/-0.3 mm,
lid height +/-0.4 mm. It is a finite sensitivity test, not joint real calibration.

Sixteen training frames supply **3,171 pixels**; selection is frames 3/9/15/21;
final test is 5/11/17/23. Frame 11 is closed. The common oracle geometry-only
interior mask has 657 test pixels: 579 iris and 78 pupil. Full-visible reporting
also includes all 1,194 visible test pixels. Training shares the eroded/boundary-
excluded mask, leaving unobserved boundary regions. No candidate-specific scoring
exclusions are allowed. Source RGB never exceeds one.

## 3. Renderer bug found before accepting scores

`scene.render.filter_size=.01` does **not** update `scene.cycles.filter_width`.
The inherited round-20 setting left Cycles at 1.5 pixels and blurred away fine
texture. `verify_r21_filter.py` proves the property separation in factory startup.
The new renderer sets and asserts the Cycles width at the frozen intended .01.
AGENTS now records this trap. Round 20's frozen files remain untouched.

The first texture check failed at frame 0 clean MAE **4.509996** display levels.
After correction it is **0.009616**. Seven open signed/narrow checks now give clean
MAE 0.007497..0.009616 and source MAE 0.013691..0.063915; closure is exactly zero.
No tolerance, iteration limit, fixture texture or split was retuned.

All renders and fits were rerun. Initial artifacts are preserved in
`composite/r21_eyes_filter_bug/`, `final/r21_eyes_filter_bug.mp4`, and four
`r21_*_filter_bug.json` reports. Their stored paths predate the archive move.
The first blurred run's apparent improvement (4.56227 source to 1.72212 selected
fit) does not describe the intended fixture and must not be cited as the result.

Blender completed both 72-image runs with actual exit code 0 and completion
markers. It warned that the external OptiX cache database could not be opened
inside the sandbox, but rendering itself succeeded on OPTIX. No cache deletion,
dependency changes, or approval escalation was needed.

## 4. Corrected result — not a repair

Mean absolute display RGB error, 0..255, fixed identical final-test masks:

| Measurement | Source/add-only | Continuous, true geometry | Split pupil, true geometry | Selected yaw -2 degrees |
|---|---:|---:|---:|---:|
| Whole scored interior | 3.97712 | 8.19838 | 6.91361 | 6.78306 |
| Iris only | 2.05177 | 5.97225 | 5.89641 | 5.87374 |
| Pupil only | 18.26914 | 24.72307 | 14.46438 | 13.53295 |
| Coupled with oracle new reflection | 3.75738 | 7.19855 | 6.33561 | 6.24558 |
| Full visible interior | 5.50438 | 11.09373 | 9.27915 | 9.76302 |
| Fit to source sum | — | 7.28313 | 7.17193 | 7.09141 |

Selection chooses a wrong supplied yaw offset: selection sum MAE 7.02839 versus
7.33004 at true geometry. **25/30 channel solves hit the 200-iteration cap without
convergence.** Only yaw+2's RGB and lid+0.4's green/blue converge. Total fitting
time is 80.39 s. Do not describe the ranking as a converged geometry optimum or
attribute it uniquely to representation error. The selected fit is unconverged.

The split pupil helps relative to the continuous representation, but all full-eye
reconstructions are worse than source. The selected output visibly smooths away
iris texture and has patchy pupil radiance. It misses four full-visible pixels;
the narrow-lid hypothesis misses 108. Boundary exclusions/unsupported coefficients
are additional limitations. No complete replacement support was earned.

New reflection in these scores is oracle target-minus-clean, not inferred scene
lighting. Tiny negative subtraction residuals are retained for scoring (minimum
-6.71e-7). The fixture's transmitted radiance is emissive; no diffuse relighting
claim is made. This failure is about this basis, design and solver budget, not
universal impossibility or the potential of a full neural inverse-eye method.

## 5. Verification, review and artifacts

- 240 saved frame/arm predictions independently sampled; max difference 2.22e-16.
- 20 whole-heldout RGB substitutions preserve fit inputs; ten access guards verify
  training-only reads. Two test substitutions and a selection access guard preserve
  all selection scores. Poisoned clean/target paths (48) leave source loading and
  selection unchanged. These are input-isolation tests, not repeated solver runs.
- All metrics, active columns, bounds and unsupported zeros recompute. Clean test
  pupil minimum is 0.00667552 linear RGB. Closed source/clean/target are exactly zero.
- 24 disabled coupled replacements and 12 inherited transaction checks pass.
- **300 exact keyed-eye fallbacks and 304 protected hashes pass again.** Existing
  real donor/context/visibility gates and production compositing remain unchanged.

`final/r21_eyes.mp4`: **96 frames, 24 fps, 960x762, four seconds**. The 24-frame
sequence loops four times, with source, known clean reference and fitted interior
above known target, experimental fit + oracle scene, and add-only control.
All 96 decoded thumbnails were inspected in four ordered sheets, along with
decoded boards 5/17 and PNG boards 0/11. Closure is intentional; lost iris detail
and patchy pupils are visible. Maximum mean encoded/input difference is 0.300162
display levels. No live playback or artist acceptance is claimed.

The real source video is reused by reference: `final/r20_source_fallback.mp4`,
unchanged hash. Original catchlights remain. Review files are display sRGB/CRF 12,
not HDR delivery. There is no new performer reconstruction to show.

Reports under the shot's `logs/`: `r21_cycles.json`, `r21_preflight.json`,
`r21_selection.json`, `r21_eyes.json`, `r21_texture_transport.json`,
`r21_filter_probe.json`, `r21_verification.json`, `r21_visual_review.json`.
The earlier round-17 work and pre-existing AGENTS additions are preserved. The
review encoder's standard latest log is left outside this round's commit.

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' -b --factory-startup --python scripts/verify_r21_filter.py -- --shot music_videos/reporter_test/shots/reporter_a
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' -b --factory-startup --python scripts/render_r21_eyes.py -- --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r21_eyes.py --shot music_videos/reporter_test/shots/reporter_a --prepare-only
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r21_eyes.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r21_eyes/review --fps 24 --out r21_eyes.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r21_eyes.py --shot music_videos/reporter_test/shots/reporter_a
```

## 6. Starter prompt for round 22

> Stay on eyes. Read round21.md and docs/r21-eye-findings.md. The independent
> textured Cycles test rejects the present low-resolution layer fit: it smooths
> away observed iris structure and is worse than source, even with true geometry.
> A separate unknown pupil grid helps only relative to the continuous grid.
> Twenty-five channel solves remain unconverged; source-sum selection chooses a
> wrong yaw hypothesis. Fix observed-detail capacity, convergence and complete
> support/uncertainty before treating geometry selection as calibration. Preserve
> same-eye identity and the joint removal/scene-reflection task; do not pivot to
> another endless tracking-only benchmark. New evaluation needs genuinely untouched
> frames/fixtures: round 21's test is now seen. Real corneal, pupil and lid geometry
> still need observation-based fitting and validation. The full EyeNeRF method
> remains unrun; handle its per-eye texture/prior issues explicitly if pursuing it.
> Set Cycles' actual filter_width and model native pixel footprints. Do not assume
> a real pupil is black, erase unobserved material, or certify layers from sum-fit.
> Keep original keyed eyes exactly wherever full replacement evidence is missing.
> Produce a labelled temporal review and round22.md. No generated performer
> texture, special capture references, key retuning or impossibility claims.
