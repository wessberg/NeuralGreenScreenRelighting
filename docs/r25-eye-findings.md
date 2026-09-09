# Partial real-eye arcs constrain image boundaries, not unique optical geometry

Round 25 fits separate-eye geometry candidates to eight newly annotated source
crops, without reading neural contours. The best training fit predicts all 24
held middle-arc points within the fixed 2px band. However, its median check error
is **.833209px**, worse than the frontal control's **.580085px**. Seventy-two to
77 of 81 tested orientations per eye remain compatible with the training arcs.
This is evidence about partial boundary constraints, not recovered anatomy or a
reflection repair. Original catchlights remain in every production eye.

## Evidence and inverse model

`scripts/data/r25_source_boundaries.json` contains source-only assistant judgments
at source 730/760/790/820, each eye separately. Native RGB, crop origins, protocol
and annotations were frozen before fitting. Each eye has nine visible limbus
points, four readable pupil points, a lid-opening polygon and a reflection
exclusion polygon. No hidden limbus was annotated. The 2px uncertainty was set
before annotation. These are new labels on reused footage, not independent human
truth or a new untouched image test. Round-24 labels were not fitted or scored.

`partial_eye_geometry.py` fits a perspective circular ring to the first and last
thirds of the visible arc (48 training points total). The contiguous middle thirds
(24 points) are checked after train-only fitting/ranking. At each signed yaw/pitch
pair on the fixed -40..40 degree, 10-degree grid, it optimizes image centre and
depth using native point-to-contour distances. Bounds and initialization use
training points only. Ring radius 5.85mm and the shot's 50mm/36mm camera are
assumptions. The ring is hypothesized to coincide with the observed transition;
finite tissue, refraction, blur and the actual limbus shape can violate that.

There is no temporal joint fit or automated boundary extraction. The numerical
module accepts arbitrary source points and camera/pixel-aspect inputs; the audit
and its selected annotations are reporter-specific experiments. It does not
impose these labels, dimensions or camera on other shots.

## Boundary checks and remaining freedom

| Check, native pixels | Best training orientation | Frontal orientation control |
|---|---:|---:|
| Median held middle-arc distance | .833209 | .580085 |
| Maximum held middle-arc distance | 1.674837 | 1.226918 |
| Held points within 2px | 24/24 | 24/24 |

The control uses the same training arcs and fits centre/depth; only its yaw/pitch
are fixed at zero. It was scored after fitting and was not substituted into the
selection procedure. This is a supplemental control, not a preregistered acceptance
criterion. Added orientation freedom gives no held-arc improvement.

593 of 648 sampled ring profiles meet the training criterion. Their held-arc
distances meet 2px on 1,569/1,779 profile-point pairs; that is a correlated family
diagnostic, not 1,779 independent observations. All eight train-compatible families
reach both ends of the sampled yaw and pitch ranges. Their topmost projected
boundary spans 9.24..13.52px, predominantly under the upper lid. This is the spread
of the sampled families, not a calibrated interval or an exhaustive bound.
Conditioned on the supplied metric radius/camera, candidate depths collectively
span 852.84..1363.37mm; these are not measured actor distances. Scaling the radius
and distance together preserves the projection exactly in the verified control.

## Connecting the candidate optics and preserving observed RGB

Each ring is placed at the actual round-24 cornea/sclera seam. The same connected
model traces through a pupil aperture to its recessed backing, retaining the
distinction between iris-plane and backing coordinates. For each compatible ring,
three corneal radii (7.2/7.8/8.4mm) and four pupil radii (1.2/1.8/2.4/3mm) are
tested. The 32 readable source pupil points condition this finite family using
a first-order native-pixel boundary residual from refracted aperture radius.
This is **in-sample pupil compatibility**, not an independent pupil test or an
exact nearest-curve error. No reflection RGB is treated as black pupil truth.

| Source / eye | Compatible rings | Compatible optical candidates | Candidate iris pixels | Candidate pupil pixels |
|---|---:|---:|---:|---:|
| 730 left | 74 | 3 | 138 | 1 |
| 730 right | 77 | 287 | 63 | 0 |
| 760 left | 73 | 0 | 0 | 0 |
| 760 right | 72 | 116 | 118 | 0 |
| 790 left | 72 | 6 | 111 | 6 |
| 790 right | 77 | 250 | 66 | 0 |
| 820 left | 73 | 4 | 149 | 4 |
| 820 right | 75 | 108 | 82 | 0 |

Of 7,116 optical candidates, 774 survive the pupil constraints. The empty family
at source-760 left abstains; it does not establish impossibility for this eye.
The coarse orientation/radius grid, assumed optical structure, ambiguous labels,
and lack of pupil decentration are possible contributors. Few surviving candidates
also do not certify a narrow uncertainty interval: missing model freedom can
make a finite family look artificially decisive.

Camera visibility comes from the observed 2D lid polygon. Candidate observations
are more than 2px plus sqrt(2) from the rasterized lid/reflection exclusion edges,
allowing for pixel-box extent and raster-centre uncertainty. A 3x3 midpoint grid
over each native one-pixel BOX must have unanimous iris/pupil classification
across every retained optical candidate. Empty families produce zero routes.

Of 2,719 conservatively eligible observed pixels, 727 receive an iris candidate
route, 11 a pupil candidate route, and 1,981 remain unknown. Per-eye archives keep
the exact source RGB, origin, native coordinates, route labels and exact fallback.
The 11 candidate pupil observations retain their nonzero RGB; none is forced black.
There is no cross-eye donor, generated texture, clean-layer estimate or reconstruction.

**These are observed sums, not clean material.** The reflection polygons are
incomplete source exclusion evidence; softer reflections may remain elsewhere.
Three-by-three sampling does not prove continuous footprint support, and the
actual camera PSF is unknown. No temporal material correspondence or decomposition
was evaluated. Candidate dots in the review must never be used as a removal mask.

The 2D lid opening does not solve lid depth, thickness or reflected-light visibility.
The connected model retains its normal discontinuity at the limbus, single effective
interface, planar centred iris, supplied globe and non-anatomical backing; posterior
cornea/lens, lashes, tear film and credible material transport remain missing.
No scene-light reflection is computed or registered to this real geometry this round.
Geometry, clean-material and production-replacement flags all remain false.

## Verification and review

The verifier checks seven signed geometry cases, connected-seam projection,
seven scale-ambiguity cases, seven asymmetric pixel-aspect cases, closure,
reflection/edge exclusions and empty-family abstention. All saved ring/pupil
scores and route archives replay. Four-times denser contour sampling changes
distances by at most .0007221px. Substituting the entire held arc leaves nine
train-only profiles exactly unchanged. RGB is absent from geometry and route
interfaces; there is no claimed hidden-RGB material test.

All **300 exact keyed-eye fallbacks and 304 protected hashes pass**. Eight new
coupled-transaction fallbacks also preserve their sources exactly. The first
verifier attempt stopped at EXR loading because OpenCV was imported before enabling
its EXR codec. Setting `OPENCV_IO_ENABLE_OPENEXR` before import fixes the verifier;
the full rerun and both encodes exit zero. No fitting thresholds changed.

- `final/r25_source.mp4`: 150 continuous frames, 60000/1001fps, 768x920, 2.5025s.
  Source left, diagnostic right, separate eyes. Fits appear at four observed
  frames only; all others explicitly abstain, including blinks. Seven decoded
  contact sheets and native diagnostic boards were inspected. Maximum mean
  encoding difference is 1.972606 display levels.
- `final/r25_detail.mp4`: 96 frames, 24fps, 768x952, 4s. The four diagnostic frames
  each hold for one labelled second so their annotations can be read. This is
  explicitly held imagery, not motion. All four decoded contact sheets were
  inspected. Maximum mean encoding difference is 1.557475 display levels.

The source review confirms original reflections remain; it is not a relighting
improvement or a continuous geometry estimate. Reviews are display-sRGB/CRF12,
not HDR masters. No live playback or artist approval is claimed.

Primary context: [EyeNeRF](https://ligethz.github.io/eyenerf_website/) separates
explicit refracting surfaces from interior/periocular representations;
[Cambridge temporal eye fitting](https://www.cl.cam.ac.uk/research/rainbow/projects/eyemodelfit/)
addresses single-camera geometry. Neither supplies validated geometry or clean
texture for this plate, and no full neural inverse method was run here.

The next useful test is joint multi-frame source evidence with explicit uncertain
lid visibility and same-eye material transport, including model/annotation
sensitivity and an independent check. Repeating independent partial-ring fits
or choosing the lowest residual would not solve the outstanding decomposition.
