# Round 25: partial source arcs and conditional inverse geometry

Frozen before new source annotations or fitting. Use separate native keyed eyes
at local frames 30/60/90/120 (source 730/760/790/820). These are reused footage,
but new boundary annotations. Round-24 labels are seen development evidence and
are not an acceptance set. Do not load neural contours in this experiment.

Inspect source-only nearest-neighbour crops, record visible limbus arcs, readable
pupil arcs, lid-opening polygons and reflection polygons. Allow 2 native pixels
of annotation uncertainty. Do not annotate hidden arcs as observed. Annotations
are assistant judgments, not independent human truth or clean-layer labels.

Fit a perspective circular limbus ring per eye using only source limbus points.
Its radius 5.85mm is a scale prior, not measured anatomy. Profile signed yaw and
pitch on a fixed -40..40 degree grid at 10 degree steps; optimize ring centre
and depth for each orientation. Use first/last thirds of the ordered visible
arc for fitting and the contiguous middle third for checking. Do not rank or
refit by check errors. This is boundary generalization within a seen source
crop, not withheld RGB, frames or independent anatomy. Report train-compatible
families (all train distances <=2px), their check errors and hidden-boundary
spread. Do not turn a low residual into a reliable geometry flag.

Connect ring candidates to round-24 cornea/sclera geometry. Profile corneal
curvature 7.2/7.8/8.4mm and pupil radii 1.2/1.8/2.4/3.0mm, without using reflection
RGB as clean pupil radiance. Readable pupil arcs may condition this finite family;
report counts before/after, and label those pupil scores in-sample. The source
lid polygon governs camera visibility with a 2px uncertainty band; it is not a
3D lid or validated visibility along reflected light rays. Ring-as-visible-limbus
and optical dimensions are hypotheses; the model lacks posterior optics and a
smooth limbus transition.

Route real observed RGB into separate-eye candidate iris/pupil observations only
where all retained geometry candidates agree at a 3x3 quadrature across the native
one-pixel BOX and the pixel footprint is conservatively inside the lid opening.
Exclude reflection polygons plus 2px. Record unknown and boundary pixels. This
finite quadrature and incomplete reflection annotation cannot certify clean
material, continuous support or the camera filter. Store original RGB and native
coordinates, no generated texture or fitted clean layer; do not mix eyes.

Test signed projection and pixel aspect, scale ambiguity, hidden-arc and RGB
isolation, closure/footprint exclusion and nonempty counts. Verify all 300 keyed
eye fallbacks and 304 protected hashes. Produce continuous 150-frame source-left
diagnostic-right review at shot fps, showing fits only at their observed frames
and explicitly abstaining elsewhere. Decode and inspect temporal contact sheets.
No production removal/replacement, lighting claim or photorealism claim is earned.

Primary context checked 2026-09-09: [EyeNeRF](https://ligethz.github.io/eyenerf_website/)
uses explicit refracting geometry alongside distinct interior/periocular models;
[Cambridge eye model fitting](https://www.cl.cam.ac.uk/research/rainbow/projects/eyemodelfit/)
addresses temporal single-camera geometry. Neither validates this partial-arc
experiment or supplies certified performer texture. No full neural method runs.
