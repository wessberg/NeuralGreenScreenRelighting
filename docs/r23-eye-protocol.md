# Round 23 protocol: residual preservation and geometry sensitivity

Frozen before new fitting or reference scoring. No production replacement is
authorized by this diagnostic. Same-eye observations only; no texture prior,
black-pupil assumption, cross-eye sharing or generated performer texture.

Twenty-four new 64x64 nonperformer Cycles frames use phase 2*pi*i/24+.47,
yaw 13*sin(phase), pitch 7*cos(phase), and lid height
4*(1-exp(-((i-17)/1.4)^2)). New texture phases, mark and environment rectangles
are specified in render_r23_eyes.py, retaining 41/79 angular frequencies and a
varying nonblack pupil. Train all except selection 2/8/14/20 and test 5/11/17/23.
The source, clean and new-environment renders are independent Cycles images.
The inherited single anterior interface, emissive interior and planar lids are
simplifications; neither the fixture nor these tests establishes real anatomy.

Use actual BOX width 1.0, 4096 samples, seed 23, OptiX, no persistent data or
denoising. Average an 8x8 midpoint quadrature of the whole pixel before fitting.
Common masks use nominal geometry on this grid plus the four footprint corners:
all subrays iris / all pupil / otherwise mixed, and union visible. Boundaries
remain in all scores. Finite quadrature does not certify continuous coverage.
Before fitting, compare direct texture transport against training frame 0 at
8x8 and 32x32; report boundary errors and do not retune against final-test RGB.
The one-pixel box is still a diagnostic filter, not the reporter's measured PSF.

Fixed 64x64 iris, 12x12 pupil, 32x32 environment; round-22 converged solver and
bounds unchanged. Four separate fits: supplied nominal geometry; corneal radius
8.1 instead of 7.8 mm; pupil radius 2.0 instead of 1.8 mm; lid height reduced by
0.15 mm (clamped at zero). These are sensitivity arms, not fitted real geometry.
Use identical nominal masks. Fit source RGB from 16 training frames only.
Selection source scores are reported, never used to choose a material certificate.

For each arm compare fitted clean, source minus fitted old reflection, and exact
source against clean truth. Preserve negative subtraction values numerically.
Report full-visible, pure iris, pure pupil and mixed-boundary scores, negative
channels, support and changes in spatial/temporal error. Couple both experimental
clean estimates to the SAME oracle target-minus-clean reflection, with add-only
control. Oracle reflection is not scene-light inference. Freeze fits and selection
scores before any final-test/reference evaluation. Residual-preserving output may
read its own source pixel, but that read must not change fitted old reflection.

On nominal training RGB, test feasibility at a fixed .002 linear per-channel
residual box (green channel) before interpreting layer ambiguity. If infeasible,
report model/data incompatibility at that tolerance, not material identifiability.
Also compute a minimal uniform residual radius using bounded linear programming;
this is model discrepancy, not measured camera noise or calibrated uncertainty.

Real geometry: fit separate corneal/pupil pose proposals to existing EllSeg
contours on six evenly spaced stored frames per eye, using assumed shot camera
intrinsics and fixed anatomical iris scale. Profile radii 7.2/7.8/8.4 mm with
independent pose/pupil optimization. Split alternating contour samples into
fit/check; explicitly these are samples of the SAME predicted ellipse, not
independent source annotations or geometry truth. Compare contour compatibility
with RITnet visible-part proposals and report reflection-ray sensitivity. No
proposal becomes validated from self-consistency; no eye RGB is replaced.

Verify algebraic residual identities, independent grid sampling, train/reference
access boundaries, LP feasibility witnesses, disabled coupled transactions,
300 exact real keyed fallbacks and 304 protected hashes. Produce a labelled
chronological review of both clean alternatives and coupled residual experiment,
plus real geometry proposal review; decode and inspect ordered contact sheets.
Record limits, artifacts and the next task in round23.md.
