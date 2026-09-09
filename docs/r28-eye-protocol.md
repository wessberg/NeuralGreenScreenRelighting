# Round 28 protocol: radiance stability and pupil-covered evidence

Freeze before running the new fits. This is a prerequisite investigation on seen
sources, not a fresh final test. Preserve round 27 and production outputs.

1. Replace the observation-column ridge with a penalty on mean squared distant
   radiance over the full sphere in the existing linear working exposure units.
   Compute the spherical-Gaussian Gram matrix analytically. Add a declared
   1e-6 coefficient penalty to make all coefficient directions coercive. Minimize
   mean RGB residual plus 1e-3 mean squared appearance coefficients and reflected
   penalties 1e-6, 1e-4, 1e-2. Keep an appearance-only control. These are fixed
   experimental priors, not measured scene brightness or calibrated SI radiance.
   Numerical column preconditioning must transform the physical penalty too.
2. Exercise an independent scalar analytic fixture: known varying Fresnel and
   angular radiance, a withheld bright lobe, and indistinguishable transmitted /
   reflected explanations. Report layer error separately from sum error. A
   stable fit is not a successful separation. Verify duplicate-observation
   invariance, exposure scaling, and analytic sphere integral by quadrature.
3. Require every reflected BOX sample to lie within 5 degrees of a training
   direction in at least two distinct fitting frames. This fixed proximity gate
   is only angular sampling coverage, not radiance or layer observability.
   Include negative and positive controls; do not count repeated pixels as frames.
4. Reuse the six round-27 geometry witnesses and observations. Add the decentered
   pupil aperture and recessed backing reachability. Fit one unknown nonnegative
   constant backing RGB per eye, shared across fitting frames, rather than
   assuming a black pupil. Keep the 17x17 source-lit appearance grid and SG basis.
   BOX samples may mix iris and backing; all must reach one of those surfaces.
   Reflected visibility remains assumed open. The inherited oversized lid shell
   is not a credible anatomical lid fit. Source-only 2px inset lid masks remain.
5. Fit sources 730/760/790/820, select on seen 805, and diagnose seen 815. Do not
   spend a fresh held source until this model can distinguish layer explanations.
   Intersect candidate and appearance/angular support across all six witnesses
   for paired scoring. Report total and labelled reflection coverage before and
   after each gate; preserve predictions even where sampling gates abstain.
   Persist both eyes' predictions before reading diagnostic 815 RGB for scoring.
6. Compare the new radiance penalty against frozen round-27 results using the
   original matrices as a separate same-support stability replay. Do not compare
   new pupil-covered scores to the old annulus-only errors as an improvement.
7. Verify full transaction fallback, protected hashes, new design replay and
   target-RGB exclusion. Encode and inspect continuous source-left temporal
   review and labelled held layer panels, including catchlights where supported.

The effective interface follows the existing reflection/Fresnel implementation;
see [PBRT](https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission),
consulted 2026-09-09. The sphere penalty, sampling gate and constant backing are
experimental choices. No clean material, physical iris-light coupling, actual
scene registration, complete removal, or production repair is implied.
