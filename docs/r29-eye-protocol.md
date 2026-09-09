# Round 29 protocol: source reflection shape and sampling

Prospective diagnostic on already seen reporter observations. No fresh final
test or production change. Freeze this file and round-28 evidence before fits.

1. Preserve the six existing geometry witnesses. Cross each with 3x3 and 7x7
   midpoint integration over one native pixel BOX. These are sampling hypotheses,
   not a measured camera PSF; no change to source colour or camera metadata.
2. Compare appearance-only, the old 53-column light, old light plus 24
   training-direction centres at concentrations 128 and 512, and the augmented
   light with a spatial 17x17 recessed backing instead of a constant backing.
   Choose the 24 centres by deterministic farthest-point sampling of reflected
   central rays in fitting observations, without RGB or reflection labels.
   Keep nonnegative fits, 1e-3 mean appearance coefficient penalty, 1e-6
   full-sphere reflected radiance penalty plus 1e-6 coefficient coercivity.
   Spatial backing is a competing source-lit appearance explanation, not material.
3. Use four leave-one-frame-out folds on source 730/760/790/820. Build each
   light dictionary from its three fitting frames only. Intersect candidate,
   two-frame appearance occupancy and five-degree/all-node angular coverage
   across every geometry, sampling and backing arm before paired fold scoring.
   A fold is eligible for reflection selection only with at least eight labelled
   reflection and eight other pixels. Require at least two eligible folds.
   Select the lowest macro-fold average of equally weighted reflection and
   other-pixel linear MSE. If this predeclared coverage fails, abstain from
   selection and retain per-fold/training diagnostics; never relax it after RGB.
   Labels and geometry are reused assistant judgments, not fresh human truth.
4. Refit all candidates on those four frames. Persist both eyes' predictions
   before reading source 805/815 RGB. Report candidate and joint support and
   sum/layer differences separately. These sources are seen diagnostics.
   Compare broad/narrow/backing on identical support, retaining failures.
5. Independently implement scalar sphere intersection, Snell/Fresnel and a
   planar pupil/iris radiance fixture; do not call production ray functions for
   fixture truth. Check signed directions, dense pixel integration, narrow
   light Gram quadrature, withheld sum/layer error and a competing material
   explanation with identical sums. Analytic fixture textures are never applied
   to performer output. Sampling proximity cannot certify layer separation.
6. Verify deterministic replay, held-fold and 805/815 RGB exclusion, saved
   evidence hashes, exact keyed fallback and disabled complete transactions.
   Encode continuous source-left eye review and labelled sparse held model
   comparisons; inspect decoded output. Report actual scope and next physical
   solve, including unresolved iris illumination, 3D lids and scene mapping.

Reflection/refraction conventions follow [PBRT's specular interface](https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission).
Pixel integration is motivated by [PBRT's image reconstruction](https://www.pbr-book.org/4ed/Sampling_and_Reconstruction/Image_Reconstruction).
Consulted 2026-09-09. Dictionary, penalties, folds and gates above are experimental
choices, not claims made by those sources. A sharper sum does not earn removal.
