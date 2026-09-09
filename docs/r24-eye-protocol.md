# Round 24: connected eye surfaces and independent visible boundaries

Frozen before new renderer output or proposal scoring. This round addresses
geometry and source evidence, not clean-material certification. No inverse texture
fit, generated performer RGB, key changes or production eye replacement.

Implement a finite anterior conic joined at an explicit limbus to an opaque
spherical sclera. Derive the cap sag from the limbus radius, and the globe centre
from the same seam. Rotate about the globe centre. An opaque iris annulus has an
actual pupil aperture; a separate recessed nonblack radiance plane is an optical
test backing, not an anatomical retina. A curved opaque shell around the globe
has a lid opening; test camera and reflected-light visibility separately.
Reject invalid dimensions. All dimensions, effective single refractive interface,
spherical sclera, seam normal discontinuity, planar iris and shell lids are
explicit assumptions. Posterior cornea, lens, tear film, lashes, SSS and diffuse
iris light transport remain absent. This is connected gross topology, not a
validated anatomical or photorealistic model.

Independently tessellate these surfaces in Cycles. Render 24 fresh nonperformer
96x96 frames at 4096 samples, seed 24, BOX width 1, no persistent data or denoising.
Gaze yaw 18*sin(2*pi*i/24+.83), pitch 10*cos(...); blink centred on 16 with
half-height 4*(1-exp(-((i-16)/1.6)^2)). Use new textured iris and nonblack backing.
Render source/clean/target using explicitly supplied finite scene emitters.
Compare analytic transport integrated over 16x16 BOX nodes against Cycles,
including mixed boundaries, pupil, iris, sclera and lids. No fitting or untouched
material acceptance claim. Scene reflection must be computed from emitters and
occluders, never obtained by subtracting target truth. A coupled oracle-material
control may use supplied old reflection; it cannot authorize real replacement.
Retain all errors, and any pre-validation corrections, in the findings.

Independent source evidence: inspect only keyed-source RGB for stored frames
15/75/135 (source 715/775/835), separately per eye, before opening their neural
contours. Record sparse visible iris/sclera transition arcs, any readable pupil
arcs, lid margins and reflection regions in native crop coordinates. Allow 2px
annotation uncertainty, omit hidden boundary portions. These are assistant
annotations on reused footage, not independent human truth or clean-layer labels.
Hash annotations and RGB provenance before scoring complete EllSeg proposals.
Do not fit or tune a model to these six cases. Report directed visible-arc
distances, proposal excursions outside the source lid opening, and pupil overlap
with observed reflection. A reflection covering a proposed pupil is unresolved
layer evidence, not proof that the pupil is wrong.

Verify surface seam continuity, nonempty visibility, no direct iris bypass,
pupil aperture/backing separation, positive/negative gaze and asymmetric pixel
aspect, lid closure and light-only occlusion, numerical invariants, independent
rendering and exact keyed-source fallback. Produce a chronological labelled
synthetic review and a continuous 150-frame separate-eye source/proposal review
at shot fps. Decode and inspect them. Record results and starter in round24.md.

Primary references checked 2026-09-09:
[EyeNeRF, SIGGRAPH 2022](https://ligethz.github.io/eyenerf_website/)
distinguishes explicit refracting eye surfaces from interior/periocular appearance;
[High-Quality Capture of Eyes](https://studios.disneyresearch.com/2014/11/19/high-quality-capture-of-eyes/)
models subject-specific eyes and pupil deformation. Neither establishes this
fixture's assumed dimensions or reconstruction from ordinary monocular footage.
