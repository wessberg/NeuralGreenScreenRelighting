# Source photometry does not yet separate the reporter's eye reflections

Round 27 fits an observed iris appearance grid and a separate reflected-light
field across source observations, comparing six geometry witnesses per eye.
**Development selection chooses the material-only control for both eyes.** The
two-layer alternatives fail to establish a transferable layer split. Some right
eye alternatives extrapolate to extreme radiance on the fresh frame. No clean
material, catchlight removal or scene-light replacement is earned.

## Frozen experiment

The protocol was written before opening source 815's native source-only boards.
The four fitting observations are source 730/760/790/820. Source 805 is development
selection; source 815 is the final numerical RGB check. Both eyes' predictions
are stored before either final RGB array is read for scoring. The source-only
annotator did see 815 RGB. This is reused footage and assistant annotation,
not independent human/anatomical truth or unseen-image acceptance.

Nine limbus marks, four pupil marks, lid opening and incomplete reflection
polygons at 815 use 2px uncertainty. The first/last limbus thirds and first two
pupil marks condition geometry; the other five marks are checks. The six best
boundary-ranked shared tuples from round 26 all remain compatible with 815.
Each uses its best boundary assignment at each observation. The complete round
26 factored family remains on disk. Six witnesses are a deliberately small
subset, not a confidence interval or a full joint geometry optimization.

For each witness, the linear observation model is the 3x3 native BOX average of
`(1-F)*A(iris_xy) + F*L(reflected_direction)`. `A` is a nonnegative bilinear 17x17
grid over +/-8mm. It represents stationary source-lit internal appearance with
a fixed transmission scale absorbed; it is **not albedo**. `L` is a constant
plus 52 nonnegative spherical Gaussians: two concentrations (8,32) at 26 fixed
cube directions. It assumes distant lighting and open reflected visibility.
The alternative material-only model fixes `L=0`.

The Fresnel weighting and reflected rays reuse the effective-interface code.
Reflection/transmission directions and dielectric Fresnel are described in
[PBRT's optical reference](https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission).
The appearance grid, light basis and stationary-appearance approximation are
experimental choices. This model does not compute iris irradiance from the
fitted environment and is not a full inverse renderer.

Nonnegative least squares uses column-normalized designs with appearance ridge
1e-3 and reflected ridge 1e-4/1e-2/1, yielding 24 models per eye. Selection uses
development linear-RGB MSE, never final RGB. All alternatives are retained.
Geometry, mapping and support accept no RGB. Fit RGB is sampled only at integer
native pixels in the four fitting frames, with full BOX candidate-iris checks
and conservative annotated lid margins. There is no donor-image interpolation
or synthesized filling of unsupported grid regions.

## Coverage and measured checks

All witnesses use identical per-frame support, intersected before fitting.
Development/final predictions additionally require each participating iris-grid
coefficient to occur in at least two distinct fitting frames. This is occupancy
support, not a conditioning, clean-material or light-direction support guarantee.

| Quantity | Left | Right |
|---|---:|---:|
| Native fitting pixels across four observations | 782 | 712 |
| Labelled reflection fitting pixels | 8 | 2 |
| Development common iris / scored pixels | 202 / 177 | 213 / 202 |
| Final conservative lid-eligible pixels | 567 | 507 |
| Final common candidate iris pixels | 189 | 174 |
| Final scored pixels | 149 | 149 |
| Final abstentions within common candidate iris | 40 | 25 |
| Final labelled reflection pixels scored | **0** | **0** |

Reflection polygons were allowed in fitting where candidate iris supported
them. Most original catchlight locations fall outside that common support,
including all final labelled catchlights. Nothing beneath them was recovered.
Unlabelled reflection, pupil/limbus errors and source shading may remain in
the fitting pixels. Repeated native pixels are correlated, not independent trials.

| Development-selected model | Left | Right |
|---|---:|---:|
| Model / geometry witness index | 16 / 4 | 8 / 2 |
| Reflected arm | disabled | disabled |
| Training linear RGB MSE | .0001145952 | .0000895312 |
| Development linear RGB MSE | .0003055847 | .0002509846 |
| Final linear RGB MSE | .0005418669 | .0031472433 |
| Final median display absolute RGB error | 7.113696 | 9.463463 |
| Final p95 display absolute RGB error | 18.149452 | 54.376019 |

Display error is per-pixel channel-mean absolute difference after sRGB encoding,
scaled by 255. Scoring does not clamp values above display white; unstable model
values can therefore exceed 255 display levels. Video previews do clamp and
explicitly report affected pixels. These are photographed-RGB prediction scores,
not clean-radiance accuracy, and differ in frame/support from round 26.

The best two-layer development alternatives are model 3 left and model 15 right,
both reflected ridge 1. Their development MSE is .0003557358/.0003356009, worse
than the selected controls. At the final check their median display errors are
6.907762/24.788167. The slightly lower left final median cannot override the
frozen development selection or establish a layer decomposition.

Selected held limbus distances are .426387/.541494/.013597px left and
.105137/.196461/.128857px right. Held pupil distances are .828505/1.574378px
left and .272118/.380036px right. All ten lie within 2px. Passing sparse boundary
checks still does not establish correct material coordinates.

## Diagnosed extrapolation failure

Post-score diagnostics, with no refit or promoted gate, show the reflected basis
can be sampled only through its tiny tails during fitting. Column normalization
then permits huge physical radiance coefficients at a modest normalized penalty.
At a new reflected direction that tail may grow many orders of magnitude.

Across the six witnesses, the largest final/training-maximum reflected-basis
ratio is 260.869836 left and **34,621,293.563824 right**. The maximum predicted
right RGB channel across all alternatives reaches **201,740.159504 linear**.
The best development two-layer right model also fails: 30/149 final pixels have
at least one channel above display white. Its review shows a bright coloured
lower rim. Nonnegativity and finite training residuals did not prevent this.

The reported all-model layer ranges include rejected, unstable alternatives.
They are not credible uncertainty bounds or evidence that every alternative is
equally plausible. Even the material-only arm can absorb photographed reflection
and lighting into its appearance grid. Its victory does not mean the eye lacks
reflections, and a better sum would not prove a clean layer split.

## Verification and review

Six signed/mixed analytic reflection cases pass (max 1.11e-16); independent
affine-grid interpolation differs by 4.44e-16 and independent scalar BOX
integration by 1.33e-15. An overdetermined positive solver fixture has max
prediction error .000431819, including ridge bias. This verifies numerics,
not recoverability of real layers. Closure and distinct-frame support checks pass.

All 48 model fits, designs, coefficients, predictions, selections and final
scores replay exactly. Two final-RGB substitution/read-guard runs per eye make
zero final RGB accesses during fitting and leave predictions unchanged. Held
pupil substitution preserves 39,690/40,824 target-table conditioning rows;
held-arc substitution preserves nine train-only profiles per eye. Source and
artifact hashes match. The first verifier run hit OpenCV's disabled EXR codec;
setting its opt-in before importing cv2 fixed the verification harness. Final
verification exits zero. **300 exact keyed-eye fallbacks, 304 protected hashes
and 12 disabled complete transactions pass.** No neural model or Cycles render ran.

`final/r27_photometry.mp4` contains 150 continuous frames, 60000/1001fps,
768x960, 2.5025s. Source left, diagnostic overlays right. Original eyes remain
through blinks; sparse marks occur at six observations, final RGB checks at 815.
Seven decoded contact sheets and the native decoded target board were inspected.

`final/r27_layers.mp4` contains four explicitly labelled one-second holds, 24fps,
960x600. The selected material-only control and the best development model for
each reflected ridge show source/sum/appearance/reflection on identical support.
All four distinct decoded holds were inspected. The small lower/side iris areas
show texture and brightness mismatches; the last right panel visibly blows out.
Black is unscored, not a black-pupil assumption or successful catchlight removal.

Maximum mean encode errors are 1.753051/.253170 display levels respectively.
These means include unchanged/black pixels and measure encoding only. Reviews
are display-sRGB CRF12, not HDR masters, live playback or artist acceptance.

The next model needs observability checks for reflected directions and physically
meaningful light regularization before trusting an inverse layer. It also needs
reflection-covered observations and explicit pupil/backing transport rather than
excluding the very area being repaired. Pupil deformation, torsion, posterior
optics, smooth limbus, true camera PSF, 3D lid/light visibility, source-lit iris
variation and actual scene registration remain unresolved. The reporter camera
and physical scale are still assumed. Complete source-reflection removal plus
actual scene-light replacement remains the task; it stays disabled here.
