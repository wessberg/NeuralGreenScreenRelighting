# Independent textured eyes do not validate the current reconstruction

The reporter's eyes remain unfixed. Round 21's corrected, independently rendered
nonperformer test is worse than retaining the observed source. No real source
removal or scene reflection was enabled. The full EyeNeRF method remains unrun.

## Experiment and information boundary

The [frozen protocol](r21-eye-protocol.md) specifies 24 Cycles frames at 64x64,
three independently rendered layers/conditions per frame, and 512x512 procedural
textures. The iris has fine angular structure and an asymmetric colour mark; the
pupil has nonzero, varying RGB. Cycles uses mesh intersections and texture sampling,
never the fitter's bilinear basis, to produce the observations. Camera conventions
and the simplified single-interface, emissive-plane anatomy remain shared.
This is independent rendering and texture-basis evidence, not real-eye validation.

Sixteen whole frames supply 3,171 training pixels. Four separate selection frames
rank nine supplied geometry hypotheses by source-sum error. Four final test frames
are untouched by fitting and selection; one is closed, leaving 657 scored pixels
(579 iris, 78 pupil). Every arm uses the same geometry-only, **oracle** scoring and
training region, eroded at the aperture and excluding a .7 mm pupil boundary band.
The supplemental full-visible region has 1,194 test pixels and includes boundaries.
This mask is not a proposed real detector, and its training exclusions leave gaps.

`eye_layer_fit.py` fits a 12x12 iris grid, an independent 12x12 pupil grid in
pupil-local coordinates, and a 12x12 environment grid. All RGB comes from source
observations. The pupil is not assumed black. A continuous-interior arm is the
control. Active scalar parameters per channel are 206 for the control and
251..286 for the split-pupil hypotheses. Bounds and iteration limits are frozen;
there is no spatial or rotational prior. Unsupported coefficients stay zero,
which is not evidence that the corresponding material is black.

Clean and target EXRs are opened only after selection is saved. The new reflection
used in the coupled score is **oracle target minus clean**, with tiny negative
Monte Carlo subtraction residuals retained numerically (minimum -6.71e-7).
It is not inferred scene lighting. The synthetic interior emits rather than
scatters light; this does not test full diffuse iris relighting.

## A renderer bug that would have flattered the model

The first implementation inherited round 20's `scene.render.filter_size=.01`.
Blender 5.2.1 actually left `scene.cycles.filter_width=1.5`. A factory-startup
probe verifies that these are separate properties. The relevant renderer setting
is the Cycles [pixel filter width](https://docs.blender.org/manual/en/4.2/render/cycles/render_settings/film.html#pixel-filter).

The first independent texture-sampling check failed: frame 0 clean display MAE
was 4.509996. After explicitly setting and asserting the Cycles width at the
already-frozen .01, it is 0.009616. Across seven open signed/narrow cases, corrected
clean MAE is 0.007497..0.009616 and source MAE 0.013691..0.063915. The eighth case
is closed. No comparison tolerance or fitting parameter was relaxed.

All 72 images were rerendered and all ten fits rerun. The initial source/fit
artifacts remain under `composite/r21_eyes_filter_bug`, the initial video under
`final/r21_eyes_filter_bug.mp4`, and four original manifests have `_filter_bug.json`
suffixes. Their original absolute paths refer to the pre-archive location; replace
`r21_eyes` with `r21_eyes_filter_bug` when locating those archived files.

The blurred first run appeared to improve selected interior MAE from source
4.56227 to fit 1.72212. That result does **not** describe the intended sharp fixture.
This is why native texture inspection and independent transport checks matter.
Round 20's constant-emitter interior checks did not expose this filtering error;
its frozen files were not modified in this round.

## Corrected results

Mean absolute display RGB errors, 0..255, on identical final-test regions:

| Arm | Selection sum | Test sum | Test interior | Iris only | Pupil only | Interior + oracle new reflection | Full-visible interior |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source / add-only control | — | — | 3.97712 | 2.05177 | 18.26914 | 3.75738 | 5.50438 |
| Continuous grid, true geometry | — | 7.28313 | 8.19838 | 5.97225 | 24.72307 | 7.19855 | 11.09373 |
| Split pupil, true geometry | 7.33004 | 7.17193 | 6.91361 | 5.89641 | 14.46438 | 6.33561 | 9.27915 |
| Yaw -2 degrees (selected) | 7.02839 | 7.09141 | 6.78306 | 5.87374 | 13.53295 | 6.24558 | 9.76302 |
| Yaw +2 degrees | 7.27517 | 7.12758 | 6.95979 | 5.84427 | 15.24039 | 6.39398 | 10.12976 |
| Radius -5% | 7.44980 | 7.18433 | 6.92015 | 5.89301 | 14.54473 | 6.38347 | 11.08347 |
| Radius +5% | 7.33188 | 7.28711 | 6.91236 | 5.90797 | 14.36804 | 6.31787 | 9.59334 |
| Pupil radius -0.3 mm | 7.39899 | 7.29210 | 7.01668 | 5.88761 | 15.39788 | 6.38189 | 11.09973 |
| Pupil radius +0.3 mm | 7.34679 | 7.18435 | 6.99126 | 5.89141 | 15.15551 | 6.41771 | 13.98865 |
| Lid height -0.4 mm | 7.41331 | 7.32238 | 6.99228 | 5.91796 | 14.96708 | 6.41802 | 17.66131 |
| Lid height +0.4 mm | 7.38311 | 7.27738 | 6.92853 | 5.92672 | 14.36500 | 6.35454 | 9.70226 |

**Twenty-five of thirty channel solves are unconverged at the 200-iteration cap.**
Only yaw+2's three channels and lid+0.4's green/blue channels report success. Total
solve time is 80.39 seconds. These are limited-budget outputs, not proven optima;
the geometry ranking is confounded by representation error and solver termination.
It cannot establish an inherent bias or uniquely explain why the wrong yaw won.

The separate pupil helps relative to the continuous grid but the eye remains worse
than source. The review visibly replaces fine iris structure with a smooth field
and leaves patchy pupil radiance. This small grid does not preserve the fixture's
detail. Full-region errors also expose excluded-boundary and unsupported-texel
problems: the selected geometry misses four visible test pixels; the narrow-lid
hypothesis misses 108. No candidate earned complete replacement support.

## Verification and review

- 240 saved frame/arm predictions match independent direct sampling to 2.22e-16.
- Twenty black/magenta whole-heldout substitutions give identical fit matrices and
  RGB; ten access guards reject any heldout read. These verify solver inputs,
  not repeated end-to-end solves.
- Two final-test substitutions and a selection-only access guard preserve the
  selected candidate and every selection score. Poisoning all 48 clean/target
  paths preserves source loading and selection; reference RGB is not a fitter input.
- All saved metrics and active columns recompute; bounds and unsupported zeros
  check. The clean test pupil minimum is 0.00667552 linear RGB, not black.
- Source, clean and target are exactly zero at closure. All 24 disabled coupled
  transactions preserve source, 12 inherited transaction checks pass, and all
  **300 keyed-eye fallbacks / 304 protected hashes** pass again.

`final/r21_eyes.mp4` is a labelled 96-frame, 24-fps, 960x762, four-second review:
24 chronological frames repeated four times. All 96 decoded thumbnails were
inspected in four ordered contact sheets; decoded boards 5/17 and PNG boards 0/11
were inspected at native board resolution. Maximum mean decoded-frame difference
from input is 0.300162 display levels. This is a display-sRGB/CRF-12 review, not
delivery. No live playback or artist acceptance is claimed. The prior real video
`final/r20_source_fallback.mp4` is reused by reference with unchanged hash.

The next experiment needs observed-detail capacity, numerical convergence and
coverage/uncertainty accounting before another geometry search can be trusted.
Preserve an untouched final-test partition; do not tune on these now-seen test
frames and call it fresh evidence. Real corneal/pupil/occlusion calibration,
same-eye material evidence and scene replacement remain outstanding. Keep the
real source, including its catchlights, whenever complete support is unearned.
