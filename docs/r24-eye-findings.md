# Connected eye transport works on supplied geometry; real layers remain unknown

Round 24 replaces the disconnected fixture topology with a finite cornea joined
to sclera, a real pupil aperture, a recessed nonblack optical backing and curved
lid occlusion. Independent Cycles images agree with the analytic forward model.
It does **not** recover real eye geometry or separate the reporter's reflection.
Production eyes retain their exact keyed source, including original catchlights.

## Geometry and the scope of the result

`connected_eye.py` derives cap sag from corneal curvature, eccentricity and
limbus radius. At the supplied 5.85 mm limbus, sag is 2.4924087106 mm; the old
2.18 mm cap did not reach that radius. The scleral sphere meets the same ring.
Both seam implicit-equation residuals are zero at float64 precision. Eye rotation
uses the globe centre, so changing gaze does not inadvertently orbit the eyeball.
The iris annulus extends to its contact with the globe at its own depth.

The pupil is an aperture in that annulus. Rays passing through it reach a separate
backing 1.4 mm deeper. Maximum observed aperture radius is 1.7999982 mm for the
1.8 mm opening; the backing hit can differ by .4776714 mm in xy. Copying the iris
plane coordinate onto a pupil texture would lose this parallax. Backing RGB is
explicitly nonblack and supplied only for this nonperformer optical test.

The opaque lid is a curved shell fixed around the globe. Camera and reflected
light rays encounter it independently. Across the 24 frames and 16x16 native
pixel quadrature, 672,209 corneal subrays see a lid along the reflected direction
while their camera ray remains open. Closed frame 16 has no visible cornea,
iris, pupil, sclera or external reflection. The lid's own emission remains visible
and reflected; the clean panel means black external environment, not removal of
all corneal reflections.

This fixes gross connectivity and visibility, **not anatomical realism**. The
cornea/sclera seam has a 15.431155-degree normal discontinuity. The model has one
effective air/interior interface, planar iris, spherical sclera and artificial
shell lids. It omits posterior cornea, lens, tear film, eyelashes, skin/iris
scattering and real lid thickness. The recessed emitter is not a recovered retina.
There is no real geometric fit or posterior distribution. Full photorealism
requires subject-specific optical geometry and material/light separation with
independent evidence; this fixture cannot meet that bar.

The primary [EyeNeRF SIGGRAPH 2022 page](https://ligethz.github.io/eyenerf_website/)
motivates distinct explicit refracting surfaces and interior/periocular appearance.
[High-Quality Capture of Eyes](https://studios.disneyresearch.com/2014/11/19/high-quality-capture-of-eyes/)
addresses detailed subject-specific shape and pupil deformation. These references
do not validate this fixture's dimensions or recovery from this plate. No full
inverse-eye neural method was run, and no capture references were required.

## Independent rendering and the two corrected bugs

`render_r24_eyes.py --shot` independently tessellates 24 fresh 96x96 nonperformer
frames, each under old, black and new external environments: 72 EXRs. OptiX,
4096 samples, seed 24, actual BOX width 1, no adaptive sampling, denoising or
persistent data. The fixture uses new texture and lighting patterns, signed gaze
and a blink. There is no inverse fit or untouched material-accuracy test.

The first run used inward mesh winding. This was caught in source inspection
before image scoring; corrected meshes assert front-facing cap normals. Original
72 images and their manifest remain in `cycles` and `r24_cycles_initial_winding.json`.
The scored images are in `cycles_oriented`.

The first analytic check on the oriented frame 0 then assigned all iris rays to
sclera: its internal sphere test included the anterior spherical patch that had
been cut out for the cornea. It had zero iris/pupil support and whole-frame clean
MAE 2.959765 display levels. Restricting internal wall hits to the finite sclera
fixes it; corrected frame-0 clean MAE is .0097801. The failed JSON and board are
preserved. The verifier reconstructs the earlier module byte-for-byte to its
render-manifest hash and confirms that geometry/texture definitions are unchanged.
See the [correction record](r24-render-orientation-correction.md).

Both render logs reach `R24_CYCLES_COMPLETE` and `Blender quit`, with all expected
files present and verified. The PowerShell wrappers returned 1 after promoting
Blender deprecation warnings on stderr to `NativeCommandError`; native Blender
exit status was not separately captured. OptiX also reported cache database
permission warnings. These are not recorded as clean process exits. No cache was
deleted or reconfigured. The analytic audit, encodes and verifier exit zero.

## Native transport scores

The analytic renderer integrates a 16x16 midpoint grid over each one-pixel BOX.
All scores include boundaries and all 24 frames. These are supplied-geometry
forward checks; they are not a comparison with round 23's inverse fit.

Weighted native-pixel display RGB MAE, 0..255:

| Region | Pixels | Analytic clean vs Cycles clean | Analytic source vs Cycles source | Coupled vs Cycles target | Add-only vs target |
|---|---:|---:|---:|---:|---:|
| Any corneal subray | 9,345 | .058440 | .155378 | .240935 | 4.522248 |
| Pure iris | 5,540 | .018219 | .068448 | .104510 | 2.162379 |
| Pure pupil | 1,140 | .004979 | .442910 | 1.124018 | 20.318341 |
| Mixed material/visibility boundary | 10,176 | .182246 | .194862 | .038443 | .699483 |
| Entire image, including background/lids | 221,184 | .009329 | .013425 | .010180 | .191065 |

The entire-image mean is diluted by unchanged background/lids; the cornea and
pupil rows matter more. Cornea and mixed masks overlap. Finite quadrature defines
these masks; it does not certify continuous footprint support. No basis rows or
coefficient-count support claims are involved.

Coupled output is `Cycles_source - supplied_old_reflection + computed_new_reflection`.
The new term is obtained from finite scene emitters and lid visibility, **never
from target-minus-clean reference RGB**. The old term, eye material and geometry
are still oracle inputs. This establishes a forward scene-reflection path for
this fixture, not production-scene registration or real source-layer inference.
Diffuse iris illumination is not changed by the external emitter in this test.

There is one negative coupled channel: frame 5, native xy (43,42), red,
-.0006166502 linear. It is retained numerically and floored only by the review
display transform. Monte Carlo rendering and finite quadrature remain in the
comparison; their contributions are not independently calibrated. Do not call
this exact material recovery or camera-noise uncertainty.

Denser 32x32 checks on frames 0/6/12/20 give corneal clean MAE .028797..030692
and source MAE .103869..145303; closed 16 has no corneal samples. This improves
integration but retains renderer discrepancy. Nothing was fitted to these images.

## Independent source observations

Six source-only native crops were inspected before opening their neural contours:
source 715/775/835, separate left/right. `r24_source_boundaries.json` stores partial
visible iris/sclera arcs, readable pupil arcs, lid openings and reflection polygons.
Annotations and source RGB/origins were frozen and hashed before scoring. They
are assistant observations on reused footage, **not independent human truth,
complete anatomical contours or certified clean radiance**. The 2px uncertainty
band was fixed before scoring. Hidden arcs remain absent.

Against EllSeg's complete contours, 24/35 observed limbus points and 9/12 pupil
points fall within 2px. Both eyes at source 775 have all seven observed limbus
points within 2px, and median pupil distances .96538/.64830px. At source 715
left and 835 left, median limbus distances are 2.48637/2.44784px. No full pupil
arc was recorded for the three least-readable cases; this is abstention.

Of 3,020 proposed eye pixels, 915 lie beyond the annotated lid opening plus its
2px band. A complete ellipse is allowed to continue behind a lid: **this count
does not itself disprove its hidden geometry**. It shows why a complete ellipse
cannot be used directly as a visible material/removal mask. The continuous review
also shows proposals during blinks. Of 148 labelled reflection pixels, 92 overlap
the proposed pupil. This is unresolved layer overlap, not proof that the proposed
pupil is misplaced. The bright pixels cannot be labelled clean pupil radiance.

No new geometry optimization was performed against these source annotations or
the predicted ellipses. No automatic boundary or material certification is earned.

## Verification and temporal reviews

`verify_r24_eyes.py --shot` passes seven signed gaze cases, seam/pivot constraints,
separate light visibility, exact closure, seven invalid-geometry rejections,
asymmetric pixel-aspect ray equivalence and analytic texture-sampling checks.
It verifies all 72 rendered hashes, replays saved image scores exactly, checks
nonempty open iris/pupil masks, runs five denser transport cases and 24 disabled
coupled transactions. All **300 exact keyed-eye fallbacks and 304 protected hashes**
pass. Original footage, key settings and production compositing remain unchanged.

`final/r24_eyes.mp4`: 72 frames at 24 fps, 864x672, three seconds. The chronological
24-frame fixture repeats three times. Six labelled panels show old scene,
black environment, analytic clean, target, coupled result and add-only. All three
decoded contact sheets and native boards 5/35/49 were inspected. The pupil retains
visible reflection differences; the simplified eye is plainly a fixture.
Maximum mean decoded-frame difference from its PNG is .592341 display levels.

`final/r24_source.mp4`: continuous 150 frames at 60000/1001 fps, 768x876, 2.5025
seconds. Source left, complete proposals right; separate eyes. Source annotations
appear only on the three specified frames. All seven decoded contact sheets and
native boards 75/135 were inspected; complete contours visibly continue over
blinking lids. Maximum mean encoding difference is 2.049835 display levels.
Lossy review pixels are not the exact-linear-source identity verification.

Both are sRGB/CRF12 reviews, not HDR masters. No live playback or artist approval
is claimed. Structured reports are in shot `logs/r24_*`; raw images, transport
arrays, failure artifacts and decoded boards are under `composite/r24_eyes` and
`composite/r24_source`. The pre-existing AGENTS/round-17 work and generic latest
review log are excluded from this round's commit.
