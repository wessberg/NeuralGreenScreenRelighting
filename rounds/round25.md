# Round 25 handoff — partial source inverse geometry; clean material unresolved

**Date:** 2026-09-09. Starts from round 24. Production defaults unchanged.
**Delivered:** eight new source-only boundary observations, separate-eye inverse
ring/optical families, contiguous withheld-arc checks, observed 2D lid exclusion,
conditional iris/pupil RGB routes, exact-source verification and two review videos.
**Not delivered:** real catchlight removal, validated anatomy, clean material,
automatic/continuous inverse tracking, production-scene coupling or photorealism.

## 1. Artist steering and scope

Stay on eyes; instrument work remains deferred. Preserve the same eye's observed
identity and generic input support. Keep original keyed eyes and catchlights until
the complete removal/replacement transaction earns support. No key retuning,
generated performer texture, capture references or tracking-only detour.

Read `docs/r25-eye-findings.md` and `docs/r25-eye-protocol.md`. No new neural model
or full neural inverse method ran. Neural contours were never loaded in this fit.
The numerical module is reusable; the audit/annotations are reporter diagnostics.

## 2. Source evidence and conditional inverse fit

Source 730/760/790/820, separate left/right eyes: nine visible limbus points and
four readable pupil points per eye, source lid polygons and incomplete reflection
exclusions. Assistant annotations at native resolution with 2px uncertainty,
frozen with RGB/origin/protocol hashes before fitting. No hidden boundary labels.
Reused footage, new labels; not independent human truth or unseen-frame acceptance.
Round-24's six annotated eyes were not fitted or scored.

`partial_eye_geometry.py` fits centre/depth of a perspective 5.85mm ring at each
of 81 signed yaw/pitch profiles (-40..40 degrees, steps of 10). The first and last
thirds of each visible arc train the fit; the middle third checks it. Initialization,
bounds and ranking use only training points. Ring radius and shot camera are priors.
The visible transition's equivalence to the physical seam is a model hypothesis.

48 training / 24 check points. Best-training orientation gives median check
**.833209px**, max 1.674837; all 24 within 2px. The same-data frontal control is
better: median **.580085px**, max 1.226918. No accuracy improvement is earned.
593/648 ring profiles are train-compatible (72..77 per eye); 1,569/1,779 correlated
family/check pairs pass 2px. Hidden top-boundary positions span 9.24..13.52px within
each family. All families reach both ends of the sampled yaw/pitch range. These
are finite sampled candidates, not a posterior or exhaustive confidence interval.
Conditional depths span 852.84..1363.37mm across cases, not measured distance.

## 3. Connected optics and observed material routes

Fit rings are placed at round-24's actual connected seam. Three corneal radii and
four pupil radii per compatible ring give 7,116 profiles. Source pupil arcs accept
774, using a first-order refracted-aperture residual in native pixels. This is
in-sample pupil conditioning, not withheld pupil accuracy. Source-760 left has
no surviving optical candidate and abstains. Do not interpret sparse or empty
families as certified anatomy or impossibility: model/grid/labels may be wrong.

| Source / eye | Ring candidates | Optical candidates | Iris routes | Pupil routes |
|---|---:|---:|---:|---:|
| 730 left | 74 | 3 | 138 | 1 |
| 730 right | 77 | 287 | 63 | 0 |
| 760 left | 73 | 0 | 0 | 0 |
| 760 right | 72 | 116 | 118 | 0 |
| 790 left | 72 | 6 | 111 | 6 |
| 790 right | 77 | 250 | 66 | 0 |
| 820 left | 73 | 4 | 149 | 4 |
| 820 right | 75 | 108 | 82 | 0 |

2D source lids govern camera visibility with 2px plus sqrt(2) raster/BOX margin;
reflection exclusions get the same margin. All retained optical candidates must
agree at 3x3 BOX nodes. Of 2,719 eligible observed pixels, 727 route to candidate
iris, 11 to candidate pupil, 1,981 remain unknown. Separate-eye archives preserve
exact original RGB/native coordinates. Pupil RGB remains nonblack. Empty families
give no support. No clean texture or reflection layer was fitted or generated.

Finite quadrature is not continuous support or the actual camera PSF. Incomplete
reflection polygons cannot certify clean radiance. 2D lid visibility is not a 3D
lid or reflected-light occlusion. Smooth limbus, posterior optics, credible lid
geometry, full material/light decomposition and real scene coupling are unresolved.
Round-24's supplied optical assumptions remain. No production repair is enabled.

## 4. Verification and review

Seven signed projection/seam cases, seven scale-ambiguity controls, seven asymmetric
pixel-aspect cases, closure/edge/reflection exclusions and empty-family controls pass.
All stored ring/pupil scores and candidate routes replay; 4x contour density differs
by at most .0007221px. Whole held-arc substitution preserves nine train-only fits.
RGB is absent from geometry/routing interfaces; this is not a material RGB holdout.
**300 exact keyed-eye fallbacks, 304 protected hashes and eight new disabled coupled
transactions pass.** First verifier attempt hit disabled EXR codec due to import
order; explicit enabling before OpenCV import fixes it. Full rerun exits zero.

- `final/r25_source.mp4`: continuous 150 frames, 60000/1001fps, 768x920, 2.5025s.
  Source left/diagnostic right, separate eyes. Four fitted frames only; others
  explicitly abstain, including blinks. Seven decoded contact sheets and native
  diagnostic boards inspected. Maximum mean encode difference 1.972606 levels.
- `final/r25_detail.mp4`: 96 frames, 24fps, 768x952, 4s. Four labelled one-second
  source-frame holds for reading annotations; not motion. All four decoded contact
  sheets inspected. Maximum mean encode difference 1.557475 levels.

Both display-sRGB/CRF12 reviews retain original reflections, not HDR masters or
artist acceptance. No live playback. Source equality is on linear arrays, not
lossy video. Reports: shot `logs/r25_*`; artifacts: `composite/r25_source` and
`composite/r25_detail`. Existing AGENTS/round-17 work and latest shared review log
are excluded from this round's commit.

```powershell
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/prepare_r25_source.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r25_source.py --shot music_videos/reporter_test/shots/reporter_a --freeze-only
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r25_source.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r25_source.py --shot music_videos/reporter_test/shots/reporter_a --prepare-detail
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r25_source/review --fps shot --out r25_source.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r25_detail/review --fps 24 --out r25_detail.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r25_source.py --shot music_videos/reporter_test/shots/reporter_a
```

## 5. Starter prompt for round 26

> Stay on eyes. Read round25.md and docs/r25-eye-findings.md. Real partial source
> arcs now drive separate-eye inverse candidates without neural contour fitting.
> Best-training orientation does worse than the frontal held-arc control; hidden
> geometry remains broad. Pupil constraints leave highly uneven finite families,
> including an empty case. Candidate iris/pupil RGB is observed reflected-plus-
> transmitted source, not clean texture. Do not promote route dots to repair masks.
> Move toward joint multi-frame source geometry/material evidence with uncertain
> lid occlusion and same-eye texture routing, rather than another independent
> partial-ring fit or supplied fixture. Account for pupil decentration, model/grid
> and annotation sensitivity, smooth limbus and posterior optics before anatomical
> claims. A narrow sampled family is not confidence. The eight new round-25 eyes
> are now seen; preserve an independent check in any extension. Keep the full
> removal plus scene-light replacement task in scope and preserve exact keyed
> source, including catchlights, until complete support is earned. No generated
> performer texture, key retuning, capture requirements or impossibility claim.
> Produce a temporal review and round26.md with measured findings and limits.
