# Round 20: observation-based eye separation and coupled replacement

Frozen before new numerical experiments, 2026-09-09. The artist's live steering
supersedes round 19's instrument priority: stay on eyes until they are fixed.

## Scope and method

Test a deterministic two-layer inverse problem, not another tracking benchmark.
Observed linear RGB is transmitted eye-interior radiance plus a Fresnel-weighted
reflected environment. Fit both from observations; no pretrained RGB decoder,
texture synthesis, smoothness loss or rotational texture prior is used.
This is an isolated experiment, not the complete published EyeNeRF method.

The conic surface uses R=7.8 mm and eccentricity 0.5, as in the inspected author
implementation of Seeing the World through Your Eyes, revision
9005f9a462a4ca5bcaeb1e101008a2e12798c915. Add Snell refraction to an interior plane
3 mm behind the apex, an explicit pupil, and a front aperture that blocks BOTH
camera and incident rays. A homogeneous n=1.376 interior is a simplification:
no posterior corneal interface, aqueous/lens interfaces, scattering or eyelashes.
The synthetic interior is emissive, so this tests transport/separation and changing
reflections, not full diffuse relighting. No anatomical values are real calibration.

## Frozen synthetic experiment

- 36 frames, 64x64, square pixels; fx=fy=3000 px, apex 1300 mm from camera.
- Yaw +/-12 degrees and pitch +/-8 degrees over a loop; independent lid closure.
- 16x16 eye-interior and reflected-environment radiance grids; bilinear sampling.
  Procedural, nonperformer texture; a black synthetic pupil. No real clean truth.
- Fit source observations on frames not congruent to 5 modulo 6. The six excluded
  frames are held out completely from fitting. Score interiors away from aperture
  and pupil boundaries. Report sample and active-parameter counts and solver status.
- Compare unknown pupil radiance with an explicitly known-black-pupil arm. That
  latter arm is an ORACLE assumption, never transferable to a photographed pupil.
- Nonnegative bounded least squares: interior [0,1], environment [0,8], tolerance
  1e-9, maximum 200 iterations, LSMR tolerance 1e-10. No result-driven retuning.
- Report total-image fit separately from recovered interior error. A good sum fit
  cannot certify either separated layer. Report untouched-source control too.
- Change the reflected environment and compare coupled reconstruction with known
  target transport, a no-removal/add-only control, and exact source fallback.
- Geometry controls: signed x/y image displacements and four signed eye tilts;
  verify conic intersections, unit vectors, reflection law, Snell law and Fresnel
  normal incidence. Include closed-aperture and incident-only occlusion cases.
- Include an exact static-ray counterexample: different nonnegative interior and
  reflection values can yield identical RGB. This is not an impossibility claim
  about other views/footage or a measurement of this performer's ambiguity.

The bilinear synthetic generator matches the fit basis deliberately: this is a
controlled algebra/transport test, not independent realistic footage validation.
Holdout frames do not make the basis independent. No synthetic result authorizes
a real repair or relaxes the existing correspondence/agreement/donor gates.

## Real source and review

Show source 950..999, both reporter eyes, alongside EXACT keyed-source fallback.
No generated performer texture, sparse-probe extrapolation, real photometric gate
score or scene reflection is allowed. Verify the existing 300 keyed-eye fallbacks
and 304 protected hashes. Preserve shot configuration and production scripts.

Write a labelled synthetic temporal comparison and a real fallback video. Synthetic
review is at 24 fps; real motion uses shot fps. Decode and inspect ordered contact
sheets and representative full boards. Record limitations, review scope and measured
results in round20.md, ending with a round-21 starter prompt focused on eyes.
