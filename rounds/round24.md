# Round 24 handoff — connected eye transport; real geometry and layers unresolved

**Date:** 2026-09-09. Starts from round 23. Production defaults unchanged.
**Delivered:** connected supplied cornea/sclera, physical pupil aperture and
recessed nonblack backing, curved dual-ray lid visibility, independent Cycles
checks, scene-computed fixture reflection, frozen source-only boundary evidence,
exact-source verification and two labelled temporal reviews.
**Not delivered:** real catchlight removal, inferred corneal/pupil/lid geometry,
complete eye material, production-scene reflection registration or photorealism.

## 1. Artist steering and scope

Stay on eyes; instrument task remains deferred. Preserve the same eye's observed
identity and generic shot inputs. Keep original keyed eyes, including catchlights,
until the complete removal/replacement transaction has earned support. No key
retuning, generated performer texture, capture references or tracking-only detour.

Read `docs/r24-eye-findings.md`, `docs/r24-eye-protocol.md` and
`docs/r24-render-orientation-correction.md`. No full inverse-eye neural method ran.
Stored EllSeg proposals were evaluated, without new neural inference or fitting.

## 2. Connected supplied model

`connected_eye.py` derives a 2.4924087106mm cap sag from the 5.85mm limbus radius
and joins the finite conic to an opaque spherical sclera. The old 2.18mm cap did
not reach this ring. Both seam implicit residuals are zero in float64. Gaze
rotates around the fixed globe centre. The iris annulus contacts the globe at
its own depth, with an actual 1.8mm pupil aperture. A separate optical backing
1.4mm deeper carries nonblack supplied RGB. Aperture/backing xy differs by up to
.4776714mm; pupil radiance is not an iris-plane texture or assumed zero.

Curved shell lids test camera and reflected-light rays separately. There are
672,209 camera-visible/light-blocked corneal subrays in the 24-frame 16x16 run.
Closed frame 16 has zero visible eye and external reflection; it still shows the
lid. The lid's own diagnostic emission and its reflection remain in clean RGB.

This is gross topology, not a validated anatomical model: the seam has a
15.431155-degree normal jump; there is one effective refractive interface,
spherical sclera, planar iris, artificial lid shell and diagnostic backing.
Posterior cornea, lens, tear film, lashes, SSS and diffuse iris transport are
absent. Dimensions are supplied assumptions, not recovered from the reporter.

## 3. Independent rendering and corrected failures

`render_r24_eyes.py` makes 24 fresh 96x96 nonperformer frames, source/black/new
environment each: 72 scored EXRs. OptiX, 4096 samples, seed 24, BOX width 1,
no denoising/adaptive sampling/persistent data. New gaze, blink, textures and
scene emitters. There is no inverse material fit or untouched acceptance test.

Initial 72 renders had inward cap/globe winding, caught before scoring. These
remain in `composite/r24_eyes/cycles`, with `r24_cycles_initial_winding.json`.
The corrected run writes `cycles_oriented` and asserts negative cap normal z.

First oriented frame-0 analytic scoring then exposed an internal sphere test
that included the removed anterior sclera patch: zero iris/pupil support and
2.959765 whole-frame clean MAE. Restricting internal wall hits to finite sclera
corrects it, yielding .0097801. Preserve `r24_transport_missing_cap_cutout.json`
and `initial_wall_failure.png`. Verification reconstructs the pre-fix module to
its recorded hash and confirms renderer parameter/texture definitions unchanged.
Both logs reach completion with all images verified, but their PowerShell wrappers
returned 1: deprecation stderr became `NativeCommandError`. Native Blender exit
status was not separately captured. OptiX cache permission warnings also remain.

## 4. Forward transport and scene coupling

Weighted native display RGB MAE (0..255), all 24 frames, 16x16 BOX quadrature:

| Region | Pixels | Analytic clean | Analytic source | Coupled vs target | Add-only vs target |
|---|---:|---:|---:|---:|---:|
| Any corneal subray | 9,345 | .058440 | .155378 | .240935 | 4.522248 |
| Pure iris | 5,540 | .018219 | .068448 | .104510 | 2.162379 |
| Pure pupil | 1,140 | .004979 | .442910 | 1.124018 | 20.318341 |
| Mixed boundary | 10,176 | .182246 | .194862 | .038443 | .699483 |

Clean/source predictions are compared with their corresponding independent
Cycles render. Entire-image clean/source MAE is .009329/.013425, heavily diluted
by background/lids. Cornea and mixed masks overlap. Finite-node masks do not
certify continuous footprints; no empty basis rows count as support.

Coupled is `Cycles_source - supplied_old + computed_new`. New reflection comes
from scene emitters and lid geometry, **not target-minus-clean truth**. Old
reflection, geometry and material remain supplied; no real source decomposition
or production-scene coupling is earned. External lights do not relight the
emissive iris diffusely. One raw coupled channel goes negative: frame 5,
xy (43,42), red = -.0006166502. Display floors it; numeric scores retain it.
Monte Carlo/finite-quadrature errors are not calibrated camera noise.

32x32 checks on open 0/6/12/20 give corneal clean MAE .028797..030692 and source
.103869..145303; closed 16 has no corneal samples. This is forward validation on
supplied data, not an inverse improvement over round 23 or a real-eye result.

## 5. Source boundaries, verification and reviews

Six source-only crops at 715/775/835 were inspected independently of their stored
neural contours, then native annotations/RGB hashes frozen. Separate eyes; 2px
uncertainty; visible arcs only. Assistant observations on reused footage are not
independent human truth, hidden material or complete anatomical geometry.

24/35 visible limbus and 9/12 pupil points fall within 2px of EllSeg's contours.
Both source-775 eyes have all seven limbus points inside that tolerance. Left
715/835 median limbus errors are 2.48637/2.44784px. Three cases abstain from pupil
arcs. Of 3,020 proposed eye pixels, 915 are outside the annotated opening plus
2px; **complete ellipses may legitimately continue behind lids**, so this rejects
their use as visibility masks, not automatically their hidden geometry. Of 148
labelled reflection pixels, 92 overlap proposed pupils: underlying material is
unresolved, not proven wrong geometry. No geometry was fit to these labels.

Verifier passes seven gaze directions, seam/pivot constraints, closure, light-only
occlusion, asymmetric pixel-aspect equivalence, seven invalid-geometry controls,
texture interpolation, all 72 render hashes, exact score replays, five denser
checks, nonempty open iris/pupil masks and 24 disabled coupled transactions.
**300 exact keyed-eye fallbacks and 304 protected hashes pass.**

- `final/r24_eyes.mp4`: 72 frames, 24fps, 864x672, 3s. Chronological 24-frame
  fixture repeated three times; six labelled panels. All three decoded contact
  sheets and native 5/35/49 inspected. Maximum mean encode difference .592341.
- `final/r24_source.mp4`: continuous 150 frames at 60000/1001fps, 768x876,
  2.5025s. Source left/proposals right, separate eyes. All seven contact sheets
  and native 75/135 inspected. Original reflections remain; complete contours
  continue over blinks. Maximum mean encode difference 2.049835.

These are display-sRGB/CRF12 reviews, not HDR masters or artist acceptance.
No live playback was performed. Source equality is on keyed linear arrays, not
the lossy reviews. Audit/verifier/encode processes exited zero; rendering has the
wrapper-status caveat above. Existing AGENTS/round-17 work and the latest review log are excluded
from this round's commit. Reports: shot `logs/r24_*`; raw artifacts:
`composite/r24_eyes` and `composite/r24_source`.

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' -b --factory-startup --python scripts/render_r24_eyes.py -- --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r24_source.py --shot music_videos/reporter_test/shots/reporter_a --freeze-only
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r24_source.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r24_eyes.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r24_eyes/review --fps 24 --out r24_eyes.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r24_source/review --fps shot --out r24_source.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r24_eyes.py --shot music_videos/reporter_test/shots/reporter_a
```

The provenance verifier accepts the current module hash on fresh rerenders;
for this round's pre-wall-fix render it reconstructs the recorded original hash
and confirms that the renderer's geometry and texture definitions are unchanged.

## 6. Starter prompt for round 25

> Stay on eyes. Read round24.md and docs/r24-eye-findings.md. Connected forward
> geometry now prevents cap bypass and distinguishes a pupil aperture from its
> backing. Fixture reflections are computed from scene geometry, but real eye
> geometry, old reflection and complete material remain uninferred. Do not
> spend another round only rendering a supplied toy or fitting the same neural
> ellipse samples. Move toward separate-eye inverse geometry/material evidence
> from source observations, using partial visible arcs and actual lid occlusion;
> account for hidden boundaries rather than completing them as truth. The six
> round-24 annotated eyes are now seen; do not tune to them as fresh acceptance.
> A complete ellipse extending behind a lid is not itself wrong geometry, and
> reflection overlap is not clean pupil material. The model still needs smooth
> limbus transition, posterior optics and credible lid geometry before anatomical
> claims. If pursuing a full neural inverse method, address same-eye texture
> routing, priors, nonblack pupil radiance and scene coupling explicitly. Keep
> exact keyed source, including catchlights, until the whole removal/replacement
> is supported. Preserve native boundaries/filter modeling and nonempty support.
> No generated performer texture, key retuning, capture requirements,
> tracking-only detour or impossibility claim. Produce a temporal review and
> round25.md with measured findings and honest remaining limits.
