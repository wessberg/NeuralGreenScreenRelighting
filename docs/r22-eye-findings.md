# Converged eye fitting still does not earn replacement

The real reporter eyes remain unfixed. Round 22 improves numerical convergence
and representation capacity on fresh nonperformer Cycles observations, but the
selected reconstruction still damages iris detail and misestimates pupil radiance.
Production defaults and original keyed eyes, including catchlights, are unchanged.

## Fresh evaluation and information boundary

The [protocol](r22-eye-protocol.md) freezes 36 new 64x64 frames: 24 train, six
select resolution, six final test. Gaze phases, texture phases, colour-mark location
and environment rectangles differ from round 21. Angular frequencies remain
41/79, the pupil has nonzero varying RGB, and textures are independently sampled
by Cycles at 512x512 resolution. Round 21's seen final-test RGB is not input.

There are 9,059 training pixels and 1,987 final-test visible pixels, comprising
1,661 iris, 323 pupil and three mixed-boundary pixels. Test frame 17 is closed.
Every arm shares the same geometry-only mask and includes boundaries in training
and scoring. Geometry is supplied true fixture geometry, not fitted real anatomy.
Four rays per pixel approximate the deliberately tiny .01 filter; their union
defines the common visible mask. This is not exact continuous-footprint coverage.

Three new arms fit iris grids 12/32/64, with separate unknown 12x12 pupil and
32x32 source environment. The inherited solver with 12/12/12 grids is a control.
RGB comes only from the same fixture's training observations, with bounds
interior/pupil [0,1], environment [0,8], and no spatial or rotational prior.
Zero-support coefficients are placeholders, never certified black material.

`eye_detail_fit.py` normalizes matrix columns and solves bounded sparse least
squares, then independently recomputes the normalized projected gradient. The
threshold is 1e-7. All nine corrected new channel solves pass at 5.17e-9..9.89e-9.
The three legacy solves all reach 200 iterations and fail at 4.32e-4..8.84e-4.
The finest arm uses 3,000 active scalar parameters per channel and converges in
508..543 iterations. Its three channels take 0.729 seconds together; all three
new arms total 0.835 seconds, excluding transport construction, LPs and rendering.
The legacy control takes 17.703 seconds. These are local timings, not end-to-end
production throughput or an isolated solver comparison with identical grids.

Source-only selection chooses the fine arm (MAE 4.67006 versus medium 6.10183
and coarse 6.42649). Selection and uncertainty are written before loading final
test source or reference RGB. This is resolution selection, not geometry calibration.

## A second filter trap, caught before final-test scoring

The first run set BOX and asserted `cycles.filter_width=.01`. Cycles' Blender
sync actually forces BOX to width 1.0; the installed UI hides the width in that
mode. The [primary implementation](https://raw.githubusercontent.com/blender/blender/main/intern/cycles/blender/sync.cpp)
explains the override. Training-frame-0 direct texture lookup failed with iris
clean MAE 5.17668 display levels, exposing the mismatch before final-test access.

The protocol was amended to BLACKMAN_HARRIS .01 and its corresponding four-ray
second-moment quadrature, offset +/-0.002770146 native pixels on each axis. No
textures, splits, capacities, solver budgets, uncertainty threshold or scoring
exclusions were changed. The [Cycles window implementation](https://raw.githubusercontent.com/blender/blender/main/intern/cycles/scene/film.cpp)
defines that footprint. The 108 EXRs and all fits were rerun. The corrected
training-frame-0 full-visible clean/source errors are 0.009923/0.023730 display
levels. Across six open verification frames they are 0.009923..0.087602 and
0.023730..0.101812. Larger boundary discrepancies expose the approximate quadrature;
these small means do not certify every boundary pixel.

The initial run is preserved in `composite/r22_eyes_box_bug/`, including its code
snapshots in `code/`, four `_box_bug.json` reports and the original protocol.
Stored absolute paths predate the archive move. It was never final-test scored
or presented as an accepted reconstruction. One coarse channel in that run returned
optimizer `success=True` despite a recomputed gradient of 6.77e-5, so numerical
certification rejected it. This supports checking the gradient instead of trusting
the [optimizer's termination flag](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).

The corrected render's actual Blender subprocess exit is zero with all 108 EXRs
and completion markers checked. The first PowerShell wrapper returned one after
native stderr warnings, despite all images and the completion marker being present;
its actual Blender return code was not captured. The corrected run uses subprocess
capture to distinguish these. OptiX cache warnings persist, but rendering completes.

## Corrected final-test result

Display RGB MAE, 0..255; identical final-test masks for every arm:

| Arm | Source sum | Clean whole visible | Clean iris | Clean pupil | Clean mixed (3 px) | Coupled with oracle new reflection |
|---|---:|---:|---:|---:|---:|---:|
| Source / add-only | — | 5.18158 | 2.25413 | 20.12986 | 16.58539 | 4.75351 |
| Legacy 12/12/12, unconverged | 6.35949 | 6.04646 | 5.83694 | 6.97598 | 21.97455 | 5.71251 |
| Coarse 12/12/32 | 6.15970 | 6.65589 | 5.87006 | 10.55716 | 21.70524 | 6.15062 |
| Medium 32/12/32 | 5.47795 | 5.98251 | 5.05066 | 10.60898 | 23.80044 | 5.48161 |
| Fine 64/12/32, selected | 3.98431 | 4.51931 | 3.18177 | 11.25186 | 20.19733 | 4.01775 |

The fine grid improves whole-eye average error relative to source, but its
iris error remains worse than source and its pupil is visibly too dark/patchy.
Increasing iris capacity does not consistently improve the separately fitted
pupil or old-reflection estimate: selected old-reflection MAE is 5.42734, versus
5.32570 for coarse. Better source-sum fit does not establish correct layers.
The joint removal/replacement task remains unsatisfied.

New reflections in the scores are oracle `target-clean` from independent renders,
not illumination inferred from a Blender production scene. Raw negative subtraction
residuals are retained (minimum -5.81145e-7). Only the disabled transaction's
display payload is clamped. Interior radiance is emissive, so this is not a diffuse
iris relighting test. Anatomy remains one interface, planar interior/lids, fixed
pupil, no scattering/eyelashes; no real performer or full EyeNeRF model ran.

## Support and conditional uncertainty

Complete coefficient footprints with at least three distinct training-frame
observations cover 1,938/1,987 visible test pixels for the fine arm: 49 fail.
Legacy/coarse/medium coverage is 1,981/1,963/1,955. More resolution increases the
observations needed. This support count alone does not establish correspondence,
visibility, clean material, independence of individual rays or real-eye geometry.

Twelve geometry-chosen final-test probes (seven iris, five pupil) have green-channel
transmitted-radiance interval widths 0.102396..0.292420 linear RGB. All reject the
frozen .01-width criterion. Each bound is a linear-program extremum over bounded
coefficients fitting the training RGB within a common residual box; all 24 saved
witnesses satisfy the constraints within 9.13e-8.

**These are conditional model intervals, not calibrated confidence intervals.**
Their box radius is 0.0902165, set by the maximum training residual plus 1e-6,
well above the .002 floor. A single large residual broadens every training
constraint. Thus these rejections do not prove ambiguity at camera-noise precision
or universal inability to reconstruct an iris. Residual modeling, boundary
quadrature, representation error and model ambiguity remain confounded. Sparse
probes cannot certify a complete eye even if all were narrow.

## Verification and review

Independent direct texture-grid sampling checks all 144 frame/arm predictions
to 1.11e-16. Per-coefficient support, normalized gradients and reported arm metrics
recompute. Eight heldout substitutions, four training access guards, two final-test
selection substitutions, a selection guard and 78 poisoned actual RGB paths verify
input separation. These check identical solver inputs, not repeated full solves.
Analytic full-rank, ambiguous-sum and unobserved-coefficient controls pass.

All 36 disabled coupled transactions, 12 inherited transaction checks, **300 exact
keyed-eye fallbacks and 304 protected hashes** pass. Closed source/clean/target
are zero; the minimum clean test pupil RGB is 0.00593564, not black.

`final/r22_eyes.mp4` is a labelled 108-frame, 24-fps, 960x762, 4.5-second review,
with the 36-frame sequence repeated three times. All decoded thumbnails were
inspected in five ordered sheets, plus decoded boards 5/23 at native board size.
Maximum mean encoding difference is 0.287007 display levels. Source, clean truth,
selected fit, target truth, coupled experiment and add-only control are labelled.
There is no live playback or artist acceptance claim. The unchanged real-source
review `final/r20_source_fallback.mp4` is reused by reference and hash-checked.

Numerical convergence is now usable for these new arms. Real observation-based
geometry, faithful observed iris/pupil detail, complete support and uncertainty,
and coupled scene replacement remain outstanding. The present test is now seen;
future acceptance needs untouched data, not retuning against these scores.
