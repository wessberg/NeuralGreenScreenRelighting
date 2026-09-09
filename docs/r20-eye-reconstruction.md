# Eye reconstruction: what round 20 establishes

The artist redirected the session back to eyes on 2026-09-09. Eyes remain the
priority until fixed; the instrument evaluation proposed in round 19 is deferred.
**The reporter's original catchlights are not fixed.** No real replacement is enabled.

## Published method inspected

[Seeing the World through Your Eyes](https://world-from-eyes.github.io/) jointly
optimizes iris texture, corneal pose and the reflected scene. Its stated primary
output is the reflected scene; an iris texture prior assists that decomposition.
This is a different problem formulation from our prior donor/track tests.

The [author code](https://github.com/HadiZayer/eyenerf/tree/9005f9a462a4ca5bcaeb1e101008a2e12798c915)
was cloned to `dependencies/eyenerf`. Inspection found details that matter to
identity-preserving use: `get_iris_texture()` routes both sides through the left
texture network; texture outputs are added to reflected RGB; rotational
regularization acts on the fitted texture; saved previews subtract the texture's
minimum. These are implementation facts, not evidence that the method cannot work.
Do not treat its saved texture preview as certified original iris radiance.

The full neural implementation was **not installed or trained**. Only three pure
geometry functions from its unchanged source were executed for conic-intersection
and normal parity checks. No released pretrained eye-reconstruction model was run.

## Implemented experiment

`corneal_transport.py` follows camera rays through a supplied anterior conic,
computes exact dielectric Fresnel reflection and Snell refraction, then intersects
an interior plane. A separate pupil region and a planar lid aperture are explicit.
The aperture is tested for camera rays **and** incident reflected rays.

`observation_eye_fit.py` fits two 16x16 radiance grids to observations with sparse,
bounded least squares. There is no pretrained decoder, spatial/rotational
regularizer or clean RGB input to the fitter. The interior/reflection split is
still conditional on geometry, bounds, sampling basis and pupil assumptions.

For a visible sample the simplified image equation is

`RGB = (1 - Fresnel) * interior(material_coordinate)
       + Fresnel * environment(reflected_ray_coordinate)`.

The reflected term is zero when its ray is blocked. The known-black-pupil arm
also sets the transmitted term to zero on its supplied pupil mask. The unknown
arm fits pupil radiance with the same continuous interior grid as the iris.
Thus the comparison includes a pupil-boundary representation mismatch as well
as uncertainty about pupil radiance; it is not a pure one-parameter ablation.

This is a deliberately restricted model: a homogeneous n=1.376 interior, no back
corneal surface/aqueous/lens interfaces, a planar iris and lids, fixed pupil size,
no lashes, no scattering, and an emissive interior. Replacement changes reflected
scene radiance; **it does not implement full diffuse iris relighting**.

The synthetic generator uses the exact same bilinear basis as the fitter. These
are controlled algebra and transport results, not independent realistic-footage
reconstruction. Six whole frames are held out; one is fully closed and contributes
zero scored pixels. No clean-donor-coverage claim or uncertainty certificate is earned.

## Numerical results

36 synthetic frames supply 13,181 visible samples. The 30 fitting frames supply
11,183; there are 404 active scalar parameters per channel in the unknown-pupil
arm and 400 in the known-black-pupil arm. Six held-out frames contain 1,094 scored
interior pixels after the frozen boundary exclusions, of which 967 are iris pixels.
The remaining 127 are pupil pixels. Source linear RGB never exceeds one.

All errors below are mean absolute display RGB differences on the 0..255 scale.
The iris-only breakdown is a supplemental material-region report computed during
verification; it does not replace the frozen whole-interior score.

| Held-out measurement | Source/add-only control | Unknown pupil | Oracle black pupil |
|---|---:|---:|---:|
| Whole interior, including pupil | 6.61804 | 1.78639 | 0.00000191 |
| Iris only | 1.80230 | 0.59517 | 0.00000216 |
| New reflection plus reconstructed interior | 6.05533 | 1.35250 | 0.00000189 |
| Sum fit to original RGB | n/a | 1.05492 | 0.00000373 |

Three of the six channel solves reached the frozen 200-iteration cap: unknown-pupil
red and oracle red/blue. They are reported as unconverged; iterations and tolerances
were not increased after results. Final fit times were 36.27 and 52.49 seconds.
Small residuals in the oracle case do not remove these status flags or certify
uniqueness. The unknown-pupil preview visibly blurs the pupil rim.

An additional exact static-ray example changes interior radiance by 0.03364 on
average while changing recombined RGB by at most 5.56e-17. This demonstrates why
sum-fit accuracy alone is insufficient. It is not an impossibility result for
moving eyes or a measurement of the reporter's identifiability.

## Independent transport checks and a bug caught

Blender 5.2.1/Cycles rendered an independent tessellated conic, interior emitter,
black pupil and opaque aperture against a constant-radiance environment plane.
Seven cases cover front, +/-yaw, +/-pitch, closed and narrow apertures. OPTIX,
4096 samples, no denoising and no persistent data were used.

The first mesh reversed the ring faces while keeping the apex fan correct. The
frontal check did not expose this clearly; +yaw had linear MAE 0.02261 and display
MAE 10.20470. Correcting the winding restores the entering-interface convention;
an explicit face-normal assertion now guards it. The same numerical tolerance
was retained. Corrected nonboundary linear MAEs range 4.24e-5..6.04e-5; display
MAEs 0.02158..0.04754, with p95 no higher than 0.11742. Closed is exactly zero.
These scores exclude material, aperture and reflected-visibility boundaries;
they do not certify antialiasing or real eyelid geometry.

There are 72..230 camera-visible samples with the incident ray blocked per open
case. A camera-space eye-opening mask alone cannot implement this distinction.
Unit vectors, the reflection law, Snell law, normal-incidence Fresnel 0.0250428,
four signed image displacements and four signed eye tilts pass. Author conic
geometry parity is within 2.51e-11 mm for points and 1.13e-12 for normals.

## Coupled replacement and what remains

`coupled_eye_relight.py` is an isolated array-level transaction. Complete clean
transmitted radiance and scene reflection share one temporal weight. Any failed
geometry, correspondence, full-reflection-extent or scene-visibility gate keeps
the source exactly. Partial support, missing data or nonfinite radiance also falls
back. Boolean evidence masks are required; NaN-valued masks cannot masquerade as
support. Twelve transaction checks pass. It is not wired into production.

All 300 existing keyed-eye fallbacks and 304 protected hashes pass. The real review
uses 100 native source crops, 950..999 for both eyes, and preserves every value
before review encoding. It earns no real repair, geometry validation, pupil
validation, material confidence or scene-replacement authorization.

Next work should fit and validate **real corneal/pupil geometry and the two observed
layers together**, with separate eyes and untouched held-out native observations.
Do not assume the photographed pupil is black. Challenge the model on independently
rendered textured eyes and geometry perturbations before reading another near-zero
same-basis residual as progress toward photorealism. Retain original source eyes
whenever the full removal/replacement transaction cannot be supported. Do not
switch back to instruments without new artist steering.
