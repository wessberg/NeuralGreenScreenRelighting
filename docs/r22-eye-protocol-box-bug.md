# Round 22 frozen protocol: detail, convergence and ambiguity

Frozen before new rendering/fitting. This is a nonperformer experiment with supplied
geometry, not real-eye calibration or permission to replace performer pixels.

36 fresh 64x64 Cycles frames, phase 2*pi*i/36 + .19, yaw 13*sin(phase), pitch
7*cos(phase), lid half-height 4*(1-exp(-((i-17)/1.4)^2)). Other optical geometry
inherits round 21. New texture phases, colour-mark location and environment
rectangles preserve the 41/79 angular-frequency challenge and nonblack pupil.
No old final-test RGB is used. Selection frames 2/8/14/20/26/32; final test
5/11/17/23/29/35; other 24 frames train. Frame 17 is closed.

4096 samples, seed 22, OPTIX, no denoising/persistent data. Explicit Cycles BOX
filter width .01, a deliberately near-point diagnostic footprint, not a realistic
camera PSF. The fitter averages four rays at (+/- .0025, +/- .0025) native pixels.
Train on every oracle visible interior pixel (including material boundaries),
using one common mask: any of the four subrays sees interior. No RGB-dependent
exclusions. Report full visible, iris, pupil and mixed-boundary pixels separately.
Simplified single-interface/emissive-interior/planar-lid anatomy remains shared.

Compare iris grids 12, 32 and 64 with independent 12x12 unknown pupil and 32x32
unknown source environment. All coefficients come only from training source RGB;
no clean texture, black-pupil assumption or spatial/rotational prior. Bounds are
interior/pupil [0,1], environment [0,8]. Also run the inherited 12/12/12 solver
as a control. Supplied true geometry is fixed; no geometry search this round.

New solver: column-normalized sparse bounded least squares via L-BFGS-B,
maxiter 20000, maxfun 50000, maxcor 30, ftol 0, gtol 1e-8. Independently recompute
projected gradient in normalized coordinates; certification requires <=1e-7.
Any cap/failure stays reported. Never retune after final-test scoring. Source-sum
selection ranks only certified new arms and is not layer/material certification.

Count support per coefficient by distinct training frames, and report test pixels
whose complete nonzero clean and reflection basis footprint has >=3 frames of
support. Unsupported values remain numerical placeholders, not black material.
Before opening final-test or clean/target RGB, freeze source-only selection and
probe uncertainty at 12 geometry-chosen visible test locations (two per open test
frame plus two additional on the first). For selected fit, bound green transmitted
radiance with two sparse linear programs per probe, with coefficient bounds and
training residual boxes centered on observed RGB. Box radius is max(.002,
maximum absolute selected training residual + 1e-6). Report this radius explicitly:
these are conditional feasible-model intervals, not calibrated confidence intervals.
An uncertainty width >.01 linear rejects that probe; partial probes never certify
an entire eye. Validate feasible LP witnesses against bounds and residual boxes.

Save fit/selection/uncertainty before loading test/reference RGB. Score sum, clean,
old reflection and clean plus oracle target-minus-clean against the same masks,
with untouched-source/add-only controls. Oracle new reflection is not inferred
scene illumination. Preserve negative reference-subtraction residuals numerically.
Real geometry and complete uncertainty are unvalidated, so every coupled production
transaction must retain the exact source. Recheck 300 real keyed-eye fallbacks
and protected hashes, input isolation, independent direct basis sampling, KKT,
support counts, LP witnesses and render transport. Hash protocol/code/artifacts.

Create chronological labelled review, 36 frames repeated three times at 24 fps,
with source, clean truth, selected reconstruction, target, coupled experiment and
add-only control. Decode all frames and inspect contact sheets/native boards.
Write findings and round22.md ending with round-23 starter prompt.
