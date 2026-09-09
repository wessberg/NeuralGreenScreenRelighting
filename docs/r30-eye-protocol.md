# Round 30 protocol: continuous source-light and optical corrections

Freeze before real fits. Seen development only; source 805/815 remain seen.
No fresh final test, production repair or generated performer texture.

1. Start from round-27 geometry witness 0 and the round-29 native candidate
   coordinates. Preserve round-29 evidence. Use one shared correction per eye:
   projected globe translation at most one native pixel on each axis, yaw and
   pitch at most five degrees. These corrections transfer to held observations;
   no held RGB adjusts its pose. This is local calibration, not continuous tracking.
2. Keep the 53 broad SG columns; add one SG initialized at the central reflected
   direction of the brightest fitting RGB sample (mean RGB, stable first tie).
   Its tangent-axis displacement is bounded to +/-20 degrees on each axis;
   concentration ranges 128..8192, initialized at 512. No known light axis is
   supplied. A single localized lobe is a restricted source-shape hypothesis.
3. Four arms: fixed seed light/fixed geometry; optimized light/fixed geometry;
   optimized light/shared geometry correction; and the latter with spatial pupil
   backing. All retain the 17x17 source-lit iris grid. Constant backing has one
   coefficient; spatial backing has 17x17. It is an appearance competitor, not
   physical material. Preserve the round-29 1e-3 mean appearance and 1e-6
   full-sphere radiance penalties, cross terms and 1e-6 coefficient coercivity.
4. Alternate exact nonnegative coefficient fitting with bounded local nonlinear
   least squares, four sweeps of at most 16 nonlinear evaluations each, then
   refit coefficients. Include 1e-4 sum of squared normalized geometry corrections.
   Keep every initial training pixel in the loss; invalid transport adds a unit
   residual per invalid pixel and never silently removes its RGB. Accept a sweep
   only if the complete penalized objective does not increase. Report local
   termination and boundary hits, never global optimality. Optimize at BOX7.
5. Four leave-one-frame-out folds on 730/760/790/820. Each fold constructs its
   own initial light from fitting RGB only. Candidate coordinates are fixed before
   fitting. Score the intersection of valid transport, two-frame appearance
   occupancy and five-degree/all-node angular support across all four fitted arms,
   further intersected with the frozen round-29 evaluation mask. These adaptive
   support losses must be counted and cannot establish an improvement over round
   29. At least eight reflections/eight others per eligible fold and two eligible
   folds for selection. Equal reflection/other MSE per fold, then macro-fold mean.
6. Persist all full and fold coefficients, parameters, predictions and support;
   both eyes must be persisted before 805/815 RGB is read for scoring. Compare
   all arms on identical pixels and report training reflection failures, transferred
   corrections, competing layers and abstentions. Reused labels are assistant
   judgments. No clean radiance or geometry confidence follows from these scores.
7. Evaluate unchanged full-fit coefficients at BOX3/7/15/31. Report component and
   sum differences on common valid pixels, boundary/interior separately. This is
   numerical sampling sensitivity, not a measured camera PSF or convergence proof.
8. Verify compact transport against the existing dense implementation and the
   independent scalar spherical fixture. Test signed pose/translation conventions,
   training-only access, objective descent, deterministic fit replay, frozen
   artifacts, 300 exact keyed fallbacks and disabled complete transactions. Build
   a continuous source-left review plus explicitly labelled held comparison stills,
   inspect decoded videos and write round30.md.

Bounded local least squares follows the interface documented by
[SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
Optical conventions retain [PBRT's specular interface](https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission).
Consulted 2026-09-09; bounds, priors, optimizer budget and gates are experimental
choices. Source-driven iris illumination, credible 3D lids/light visibility,
posterior optics and actual scene-light registration/replacement remain necessary
physical work. A stationary source-lit grid cannot complete that transaction.
