# Round 27 handoff — source appearance/light fit fails held prediction

**Date:** 2026-09-09. Starts from round 26. Production defaults unchanged.
**Delivered:** source-photometric ranking of geometry witnesses, a two-layer
appearance/reflected-light inverse model, a fresh source-815 RGB check,
extrapolation diagnosis, exact-source verification and two encoded reviews.
**Not delivered:** clean iris material, catchlight removal, continuous inverse
tracking, validated anatomy/light visibility, actual scene coupling or photorealism.

## 1. Artist steering and scope

Stay on eyes; instrument work deferred. Preserve observed same-eye identity and
generic input support. Keep original keyed eyes/catchlights until complete removal
and actual scene-light replacement earn support. No key retuning, generated
performer texture, capture references or tracking-only detour. Read
`docs/r27-eye-findings.md` and the frozen `docs/r27-eye-protocol.md`.

The numerical module takes explicit observations/geometry/camera. Fixed frame
selection and source-only assistant annotations are reporter diagnostics, not
an automatic generic production estimator. No neural contours were loaded;
no neural model or Cycles render ran. Reporter camera/scale remain assumed.

## 2. Joint source appearance and light witnesses

Source 730/760/790/820 fit photometry; source 805 selects models; source 815
(local 115) is the fresh numerical final check. The protocol predates inspecting
815 source boards and fitting. New source-only annotations have nine limbus and
four pupil marks, lid opening and incomplete reflection polygons, uncertainty
2px. First/last arc thirds and first two pupil marks condition geometry; the
remaining five marks are checks. These are assistant judgments on reused footage,
not independent human truth or unseen-image acceptance.

Use six best boundary-ranked shared tuples per eye from round 26. All six accept
815's training boundaries. Best per-frame boundary assignments are used; the
complete round-26 factored family remains preserved. This small witness subset
does not enumerate geometry ambiguity or establish confidence. Observed RGB
now ranks these witnesses, but poses are not continuously photometrically fitted.

The experimental forward model BOX-integrates `(1-F)*A(iris_xy)+F*L(reflected)`:
nonnegative 17x17 bilinear source-lit iris appearance plus constant/52 spherical
Gaussian distant-light basis terms. `A` is not albedo; iris irradiance is not
computed from `L`. Stationary appearance, zero torsion and open reflected
visibility are assumptions. This is a reduced inverse appearance model, not
a complete physical material/geometry/light solve.

Each geometry has a material-only control and three reflected-ridge variants,
24 models per eye. Nonnegative least squares uses column normalization and fixed
ridge penalties. Four-frame fit support is the geometry-witness intersection:
782 left/712 right native pixels, including only eight/two labelled reflection
pixels. Prediction requires each iris-grid coefficient observed in two fitting
frames. Identical development/final support is used across all models.

## 3. Measured outcome and failure

Development RGB selects **material-only** for both eyes. Final predictions were
saved for both eyes before either final RGB array was numerically scored.

| Selected model | Left | Right |
|---|---:|---:|
| Model / geometry witness index | 16 / 4 | 8 / 2 |
| Training linear RGB MSE | .0001145952 | .0000895312 |
| Development linear RGB MSE | .0003055847 | .0002509846 |
| Final linear RGB MSE | .0005418669 | .0031472433 |
| Final median display absolute RGB error | 7.113696 | 9.463463 |
| Final p95 display absolute RGB error | 18.149452 | 54.376019 |
| Final scored native pixels | 149 | 149 |
| Final labelled catchlight pixels scored | **0** | **0** |

Final common candidate iris is 189/174 pixels; 40/25 additionally abstain for
appearance-grid support. Conservative lid-eligible counts are 567/507. Original
catchlight areas were not recovered. Incomplete masks and source shading can
contaminate the fitted appearance. The selected material-only arm is not clean
material and does not imply an absence of reflections.

Best two-layer development choices, model 3 left/15 right (ridge 1), have worse
development MSE .0003557358/.0003356009. Final median display errors are
6.907762/24.788167. The left final median does not override the frozen selection.
These checks differ from round 26's target/support and cannot be compared as
a round-over-round improvement. Display scoring extends sRGB above white without
clamping; unstable alternatives can exceed 255 display levels. Reviews clamp
and label clipping explicitly.

All ten selected held boundary checks are within 2px: limbus
.426387/.541494/.013597 left, .105137/.196461/.128857 right; pupil
.828505/1.574378 left, .272118/.380036 right. Boundary agreement again does not
establish correct material coordinates.

**Concrete model failure:** column normalization can give enormous radiance
coefficients to light lobes seen only through tiny fitting tails. New reflected
angles activate them strongly. Post-score conditioning diagnostics (no refit)
measure maximum final/training light-basis ratios 260.869836 left and
34,621,293.563824 right. One right alternative reaches 201,740.159504 linear
in a predicted channel. Even the best development two-layer right model clips
30/149 pixels in its final display. This is failed extrapolation, not credible
material ambiguity. Nonnegativity and training fit are insufficient safeguards.

No thresholds, grids or penalties were retuned using 815; it is now seen for
future rounds. All alternatives/failures are preserved. The model has no
reflected-direction observability gate, pupil/backing fit, physical iris-light
coupling or credible 3D lid/light occlusion. These limitations prevent it meeting
the photorealism bar. They do not establish universal impossibility.

## 4. Verification and review

Independent signed reflection (six cases), affine-grid, scalar BOX integration,
positive solver fixture, closure and distinct-frame support checks pass. Maximum
reflection/BOX errors are 1.11e-16/1.33e-15; the ridge-biased positive fixture
prediction error is .000431819. These are numerical controls, not clean-material
or anatomy evidence.

All 48 fits/designs/predictions/selections/scores replay exactly. Two final-RGB
substitution/read-guard runs per eye make zero target reads during fitting and
preserve predictions. Held-pupil substitution preserves 39,690/40,824 conditioning
rows; held-arc substitution preserves nine profiles per eye. Source/artifact
hashes match. Initial verification hit the disabled EXR codec; the harness now
sets its OpenEXR opt-in before cv2 import. Final verifier exits zero.

**300 exact keyed-eye fallbacks, 304 protected hashes and 12 disabled complete
transactions pass.** Original archives and production outputs remain unchanged.

- `final/r27_photometry.mp4`: 150 continuous frames, 60000/1001fps, 768x960,
  2.5025s; source left/diagnostic overlay right. Original catchlights persist
  through motion and blinks. Six sparse observation overlays, final checks at 815,
  explicit abstention elsewhere. Seven decoded contact sheets and decoded target
  inspected. Maximum mean encode error 1.753051 display levels.
- `final/r27_layers.mp4`: 96 frames, 24fps, 960x600, 4s. Four labelled one-second
  holds, explicitly not motion: selected control, then best development model
  for each reflected ridge. Source/sum/appearance/reflection share identical
  support. All four distinct decoded holds inspected. The last right panel has
  a visibly blown-out coloured lower rim; black means unscored. Maximum mean
  encode error .253170 display levels.

Reviews are display-sRGB CRF12, not HDR masters, live playback or artist approval.
Exact-source claims concern linear arrays, not lossy encodes. Artifacts are under
`composite/r27_photometry` and `composite/r27_layers`; reports are `logs/r27_*`.
Existing AGENTS/round-17 work and the generic latest review log are excluded from
this round's commit.

```powershell
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/prepare_r27_source.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r27_photometry.py --shot music_videos/reporter_test/shots/reporter_a --freeze-only
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r27_photometry.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/review_r27_photometry.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r27_photometry/review --fps shot --out r27_photometry.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r27_layers/review --fps 24 --out r27_layers.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r27_photometry.py --shot music_videos/reporter_test/shots/reporter_a
```

## 5. Starter prompt for round 28

> Stay on eyes. Read round27.md and docs/r27-eye-findings.md. The first real-source
> appearance/reflected-light witness fit loses to its material-only control on
> development RGB. Tiny sampled light-basis tails plus column normalization yield
> enormous unconstrained radiance at new angles; nonnegative fits and occupied
> iris-grid cells do not provide angular observability. Preserve the frozen failed
> result. Source 815 is now seen. Before another final check, make reflected-light
> support and radiance regularization meaningful in physical units and exercise
> extrapolation on an independent controlled fixture. Keep competing material and
> illumination explanations. Get reflection-covered evidence into the fit,
> including pupil/backing transport where needed; another annulus-only score with
> zero catchlight coverage cannot establish removal. Move toward coupled source
> material/geometry/light inference with a fresh held RGB check, credible 3D lid
> visibility and actual scene-light replacement. Do not promote an appearance grid
> to clean material or a small geometry subset to confidence. Preserve exact keyed
> source until the whole removal/replacement transaction earns support. No generated
> performer texture, key retuning, capture requirements or impossibility claim.
> Produce a temporal review and round28.md with measured findings and limits.
