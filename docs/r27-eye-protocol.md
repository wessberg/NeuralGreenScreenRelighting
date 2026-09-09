# Round 27 frozen protocol: source appearance and reflected-light competition

Written before opening source 815 native eye boards or numerical RGB scoring.
All prior footage is reused; the fresh check is computational, not unseen footage
or independent human truth. No production removal/replacement is authorized by
this diagnostic's scores.

Use source 730/760/790/820 to fit appearance, source 805 for development model
selection, and source 815 (local 115) for a fresh final RGB check. Keep eyes
separate. Source-only assistant marks at 815 specify nine limbus points, four
pupil points, lids and incomplete reflection polygons with 2px uncertainty.
Only first/last arc thirds and first two pupil marks condition target geometry.
Reserve remaining marks. Save predictions before final target RGB scoring.

Take the six best boundary-ranked shared tuples from each round-26 family,
retaining the original complete family on disk. Use the best boundary assignment
per tuple at the five existing observations. Fit 815's ring family without RGB;
retain each tuple's best compatible target assignment. This is a deliberately
limited witness search, not enumeration, a posterior or confidence. No threshold
or geometry grid tuning using final RGB.

Forward model in linear RGB: BOX average of (1-F)*A(u,v) + F*L(reflected direction).
A is a nonnegative, bilinear 17x17 iris-coordinate appearance grid over +/-8mm;
it is source-lit internal radiance with a fixed transmission scale absorbed, NOT
albedo or certified clean material. L is a nonnegative distant reflected-radiance
field: constant plus spherical Gaussians at the 26 normalized nonzero cube
directions, concentrations 8 and 32. Both layers share coefficients across
observations of one eye. A material-only arm fixes L=0. Source-light visibility
is assumed open; credible lids, near-field lighting, iris illumination changes,
pupil deformation, torsion, posterior optics and camera PSF are unmodelled.
This is a reduced inverse appearance model, not a full physical inverse renderer.

Use native integer pixel observations with 3x3 BOX quadrature, 2px plus sqrt(2)
margin inside annotated lids. Include labelled reflection areas when all BOX
samples reach candidate iris; never assume their RGB is clean. Intersect candidate
pixel support across the tested geometry witnesses before any fit or comparison.
Predict only where every participating iris-grid coefficient has observations in
at least two distinct fitting frames. Report abstention and reflection coverage.
No source RGB outside those fitting observations enters coefficients.

Nonnegative least squares uses column-normalized designs with a fixed 1e-3
appearance ridge and reflected-field ridge 1e-4, 1e-2, or 1; keep all alternatives.
Rank on development-frame linear RGB MSE on identical support across all models,
including the material-only arm; score the fresh target once, without refitting.
Also report the best development-ranked material-only control and full competing
layer/prediction ranges. A better photographed-RGB prediction cannot certify a
layer split. Report training versus development versus final errors separately.

Verify independent signed reflection controls, BOX integration, nonnegative
solver reconstruction on a controlled fixture, final-RGB exclusion/substitution,
held-label exclusion, repeatability and input/protected-output hashes. Verify all
300 keyed-eye fallbacks and complete disabled transactions. Preserve source
archives. Produce a continuous 150-frame shot-fps source-left review with sparse
diagnostic overlays/explicit abstention, plus readable held layer comparisons;
decode and inspect both. Document findings and limits in round27.md.

Optical reference: PBRT, Specular Reflection and Transmission,
https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission
(consulted 2026-09-09). The existing effective-interface Fresnel/reflected rays
are reused; the grid, light basis and stationary-appearance assumptions are
experimental choices and are not established by that reference.
