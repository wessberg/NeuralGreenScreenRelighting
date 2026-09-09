# Pre-score winding correction

After the initial 72 EXRs completed, source inspection of `ring_surface` found
radial-then-angular face order, giving inward cap/globe normals. This reverses
the air/interior dielectric orientation. No analytic image scores had been run.
Reverse face order and assert negative cap normal z for nondegenerate faces.
The corrected run writes `cycles_oriented`; initial images in `cycles` and
their original manifest `r24_cycles_initial_winding.json` remain available.
No dimensions, texture, sampling, labels or acceptance tolerances change.
The audit must verify this correction's hash in addition to the frozen protocol.

The first oriented frame-0 analytic comparison exposed a second implementation
error: the refracted wall test intersected the full globe sphere, including the
anterior patch removed for the cornea. Every iris ray was falsely assigned to
sclera: zero iris/pupil support, clean whole-frame MAE 2.959765 display levels.
`r24_transport_missing_cap_cutout.json` and `initial_wall_failure.png` preserve
this failure. Apply the same finite sclera restriction (z >= seam sag) to
internal wall intersections. The renderer's topology and all fixture parameters
are unchanged, so its manifest retains the earlier analytic-module hash; the
verifier checks rendered parameter/texture invariance explicitly. No fitting
or heldout acceptance is involved in this forward-transport diagnostic.
