# Round 22 handoff — convergence improves; eye replacement remains unearned

**Date:** 2026-09-09. Starts from round 21. Production defaults unchanged.
**Delivered:** finer observed-only layer fitting, independently checked convergence,
training-frame support and LP uncertainty diagnostics, a fresh textured Cycles
evaluation, a second filter fix, exact-source verification and labelled video.
**Not delivered:** a real reporter repair, reliable real eye geometry, faithful
complete iris/pupil material, inferred scene reflections or photorealism.

## 1. Artist steering and scope

Stay on eyes; instrument task remains deferred. Preserve same-eye observed identity,
separate eyes, generic input support and exact keyed source including catchlights
when the entire removal/replacement transaction lacks evidence. No key retuning,
generated performer texture or special capture references. The full EyeNeRF method
is still uninstalled/untrained; no pretrained neural model ran this round.

Read `docs/r22-eye-findings.md` and `docs/r22-eye-protocol.md`.

## 2. What was built and frozen

`eye_detail_fit.py` fits independent iris/pupil/environment grids from supplied
subray transport and observed RGB. It uses column normalization and bounded sparse
least squares, explicitly checks the normalized projected gradient, counts distinct
training-frame support per coefficient, and computes conditional feasible-model
intervals with two linear programs per queried transmitted colour.

`render_r22_eyes.py` independently renders 36 new 64x64 frames as source, clean
and target: 108 EXRs. New gaze/texture phases, mark position and environment
rectangles preserve round 21's 41/79 angular-frequency and nonblack-pupil challenge.
The single-interface/emissive-plane/planar-lid anatomy remains shared and simplified.

`audit_r22_eyes.py` compares 12/32/64 iris grids with separate 12x12 unknown pupil
and 32x32 environment. A 12/12/12 inherited-solver arm is a control. No geometry
search: supplied true geometry only. Twenty-four training frames yield 9,059
native observations; six selection frames choose capacity; six final-test frames
yield 1,987 visible pixels (1,661 iris, 323 pupil, three mixed). Frame 17 is closed.
Common oracle masks include boundaries in training and scoring, with no
candidate-specific exclusions. Numerical zero-support values are not material.

## 3. Filter bug caught before final-test access

BOX silently forces a one-pixel filter in Cycles' Blender sync, even when the
actual Cycles Python width property reads .01. The initial training-frame-0
texture sanity check failed (iris clean MAE 5.17668 display levels). The installed
UI and primary Blender source corroborate the override. AGENTS now records it.

The protocol was amended before rerendering/final-test scoring to BLACKMAN_HARRIS
.01 and a four-ray second-moment quadrature (offset +/-0.002770146 native pixels).
No fixture texture, split, grid size, solver budget, interval threshold or scoring
exclusion was changed. The initial training/selection fits and LP results were
seen; only frame-0 clean truth was opened for diagnosis. Final-test RGB stayed closed.

The corrected full-visible frame-0 clean/source MAEs are 0.009923/0.023730. Six open
transport checks give clean 0.009923..0.087602 and source 0.023730..0.101812.
The tiny diagnostic footprint is not a real camera PSF. Four rays do not exactly
integrate material boundaries or cover the continuous filter support.

The initial 108 EXRs, fits and code snapshots remain under
`composite/r22_eyes_box_bug/`; four `_box_bug.json` reports and
`docs/r22-eye-protocol-box-bug.md` preserve its pre-test state. Stored paths predate
the move. No initial final-test evaluation/video was produced. The corrected
Blender subprocess exited zero; `r22_render_process.json` records it. The initial
PowerShell wrapper returned one on native stderr warnings, and its actual Blender
exit was not captured. Both runs had completion markers and all images. OptiX cache
warnings did not prevent rendering. No cache changes or dependency installation.

## 4. Corrected result and limitations

Display RGB MAE, 0..255, same final-test masks:

| Arm | Source sum | Whole visible clean | Iris clean | Pupil clean | Coupled with oracle scene |
|---|---:|---:|---:|---:|---:|
| Source / add-only | — | 5.18158 | 2.25413 | 20.12986 | 4.75351 |
| Legacy 12/12/12 | 6.35949 | 6.04646 | 5.83694 | 6.97598 | 5.71251 |
| Coarse 12/12/32 | 6.15970 | 6.65589 | 5.87006 | 10.55716 | 6.15062 |
| Medium 32/12/32 | 5.47795 | 5.98251 | 5.05066 | 10.60898 | 5.48161 |
| Fine 64/12/32, selected | 3.98431 | 4.51931 | 3.18177 | 11.25186 | 4.01775 |

All nine corrected new channel solves pass the 1e-7 normalized-gradient gate
(5.17e-9..9.89e-9). All three inherited solves hit 200 iterations and fail
(4.32e-4..8.84e-4). Fine has 3,000 active parameters per channel, takes 508..543
iterations and 0.729 s for RGB; all new solves total 0.835 s, legacy 17.703 s.
These exclude transport/LP/render time and use different environment capacities.
Initial BOX coarse green returned success with gradient 6.77e-5; our gate rejected
it. Do not equate optimizer success with numerical certification.

Selection source-sum MAEs: coarse 6.42649, medium 6.10183, fine 4.67006. The selected
fine fit improves whole-eye average over source but still worsens iris-only error,
darkens/patches the pupil and changes observed structure. Mixed-boundary MAE is
20.19733 versus source 16.58539 on three pixels. A better sum does not prove layers.

Fine has complete >=3-training-frame coefficient support at 1,938/1,987 visible
test pixels; 49 fail. Twelve geometry-only probes (seven iris/five pupil) all reject
the .01 linear uncertainty-width criterion: green intervals span widths
0.102396..0.292420. All 24 LP witnesses are feasible within 9.13e-8.

**The uncertainty box radius is 0.0902165**, inherited from the largest training
residual plus 1e-6. This globally loose box broadens every constraint. Intervals
are conditional model ranges, not calibrated confidence or evidence of ambiguity
at camera-noise precision. Model error, boundary quadrature and ambiguity remain
confounded. Sparse probes never certify a complete eye. Do not claim impossibility.

New scene reflection is still oracle target-minus-clean. Its raw negative residuals
are retained (minimum -5.81145e-7). No scene illumination was inferred and emissive
interiors do not test diffuse iris relighting. No real eye replacement was enabled.

## 5. Verification, review and artifacts

- 144 saved frame/arm predictions match independent scalar sampling to 1.11e-16.
  Saved support, numerical gradients and all arm scores recompute.
- Eight heldout substitutions, four training access guards, two final-test
  selection substitutions, one selection guard and 78 poisoned actual paths pass.
  These establish identical inputs, not repeated end-to-end solves.
- Full-rank numerical, ambiguous-sum and unseen-coefficient controls pass.
  Closed source/clean/target are zero; clean test pupil minimum is 0.00593564.
- 36 disabled coupled replacements and 12 inherited transaction checks pass.
  **300 exact keyed-eye fallbacks and 304 protected hashes pass again.**

`final/r22_eyes.mp4`: 108 frames, 24 fps, 960x762, 4.5 seconds; 36 chronological
frames repeated three times. Six labelled panels show source, clean reference,
selected fit, target reference, coupled experiment and add-only. All decoded
thumbnails in five ordered sheets and native boards 5/23 were inspected. Maximum
mean encoding difference is 0.287007 display levels. Pupils remain too dark and
patchy; iris detail differs from truth. No live playback or artist acceptance.

`final/r20_source_fallback.mp4` is reused by reference and hash-checked unchanged;
real original catchlights remain. Reviews are display-sRGB/CRF12, not HDR delivery.
Structured reports under shot logs: `r22_cycles`, `r22_render_process`,
`r22_transport_pretest`, `r22_preflight`, `r22_selection`, `r22_uncertainty`,
`r22_eyes`, `r22_verification`, `r22_visual_review`, `r22_artifacts` (all JSON).
Prior round-17 work and pre-existing AGENTS changes are preserved. The encoder's
standard latest `review_video.json` is excluded from this round's commit.

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' -b --factory-startup --python scripts/render_r22_eyes.py -- --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r22_eyes.py --shot music_videos/reporter_test/shots/reporter_a --transport-only
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r22_eyes.py --shot music_videos/reporter_test/shots/reporter_a --fit-only
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r22_eyes.py --shot music_videos/reporter_test/shots/reporter_a --evaluate-only
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r22_eyes/review --fps 24 --out r22_eyes.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r22_eyes.py --shot music_videos/reporter_test/shots/reporter_a
```

## 6. Starter prompt for round 23

> Stay on eyes. Read round22.md and docs/r22-eye-findings.md. Numerical convergence
> is now verified for the new coarse/medium/fine arms; do not spend another round
> merely increasing iterations. Fine resolution improves whole-eye average on new
> Cycles data but still damages iris texture and makes the nonblack pupil too dark.
> A source-sum fit is not a material certificate. Forty-nine test pixels lack even
> three-frame coefficient support. All twelve conditional LP probes reject, but
> their global .0902 residual box is too loose to infer ambiguity at noise precision.
> Address observation-based layer/material evidence and real corneal/pupil/lid
> geometry, preserving the joint removal and scene-reflection transaction. Evaluate
> whether residual-preserving reconstruction or a full inverse-eye method can retain
> actual observed detail without assigning unresolved pupil radiance to reflection.
> Do not reuse round22's now-seen test to tune and claim fresh acceptance. Model the
> actual filter and native material boundaries; BOX overrides width to one pixel.
> Keep every real keyed eye exactly, including catchlights, until complete support,
> validated geometry and coupled replacement are earned. Full EyeNeRF remains unrun;
> handle same-eye/per-eye priors explicitly if pursuing it. No generated performer
> texture, capture references, key retuning, tracking-only detour or impossibility
> claims. Produce a labelled temporal review and round23.md.
