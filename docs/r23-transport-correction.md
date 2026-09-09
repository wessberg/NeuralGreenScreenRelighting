# Pre-fit correction: direct rays beside the finite corneal cap

After rendering, but before any fitting, selection or final-test/reference access,
the prescribed training-frame-0 check found a clean boundary MAE of 10.74118
display levels even at 32x32 integration. Increasing quadrature was not the fix.
The fixture has a finite open corneal cap over a larger emissive iris plane.
Camera rays missing the cap can hit that plane directly; the inherited trace
dropped them. `finite_cornea_transport.py` now explicitly intersects those rays
with the interior plane, retaining aperture visibility and assigning zero Fresnel
reflection. Existing shared corneal_transport.py and old round artifacts are unchanged.

This changes the common nominal geometry mask to include direct observations;
no RGB-dependent boundary exclusion is used. All four fitting arms use that same
mask. The corrected frame-0 visible mask has 483 pixels rather than 457, so these
whole-mask before/after means are not a paired-pixel accuracy comparison.
Corrected clean MAE at 32x32 is .0458051 overall and .126662 mixed-boundary;
at the frozen 8x8 fitting quadrature it is .307978 overall and 1.063463 mixed.
The finite 8x8 error is retained and reported, not hidden by tuning after testing.
The open cap with directly visible interior is a fixture simplification, not
an anatomical eye model. No renderer, texture, sample count, split, bounds,
solver, fit capacity, feasibility threshold or real-eye proposal rule changed.

`r23_transport_missing_direct.json` preserves the failed check.
`r23_transport_pretest.json` is the corrected pre-fit check. The rendered manifest
continues to hash the original protocol; fit reports additionally hash this
amendment. Final-test truth remains unopened at the time of this amendment.

The [primary Cycles filter implementation](https://raw.githubusercontent.com/blender/blender/main/intern/cycles/scene/film.cpp)
was checked: BOX width 1 integrates offsets -.5..+.5. This mismatch was missing
geometry transport, not another filter-width override.
