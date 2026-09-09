# Round 26 handoff — joint source eye families; material still unresolved

**Date:** 2026-09-09. Starts from round 25. Production defaults unchanged.
**Delivered:** shared optical constraints across five observations per eye,
pupil decentration/depth/grid/annotation sensitivity, a fresh held source-RGB
check, a corrected inverse-ray solver, exact-source verification and three reviews.
**Not delivered:** clean iris material, original catchlight removal, validated
anatomy, continuous inverse tracking, real scene-light coupling or photorealism.

## 1. Artist steering and scope

Stay on eyes; instrument work remains deferred. Preserve observed same-eye
identity and generic input support. Keep original keyed eyes/catchlights until
the complete removal plus scene-light replacement transaction earns support.
No key retuning, generated performer texture, capture references or tracking-only
detour. Read `docs/r26-eye-findings.md` and `docs/r26-eye-protocol.md`.

No neural model or new Cycles render ran. Neural contours were never loaded.
The numerical module accepts source observations and camera inputs; the frame
selection, annotations and audit are reporter diagnostics, not an automatic
generic production estimator. The reporter's shot camera remains assumed.

## 2. Joint geometry and independent checks

Reused source 730/760/790/820 donor labels/ring families; added source 805
(local 105), each eye separately. Source-only assistant labels were made after
protocol freeze, before fitting: nine limbus points, four pupil points, observed
lid opening and incomplete reflection exclusion. Native RGB/origins, archives,
labels, prior fit, camera and protocol hashes are frozen. Reused footage and
new assistant judgments are not unseen-image acceptance or human/anatomical truth.

Only source 805's first/last arc thirds and first two pupil points condition
geometry. Its middle three arc points and last two pupil points are checks.
Donor pupil marks are development conditioning. Target RGB is absent from
geometry, selection, support and donor prediction; predictions are saved before
numerical target-RGB scoring. The annotator did inspect source-only RGB.

`joint_eye_geometry.py` requires one shared corneal radius (7.2/7.8/8.4mm), iris
depth (3.3/3.6/3.9mm) and pupil xy offset ({-.6,0,.6}mm squared) across all five
observations of one eye. Per-frame pose/centre/distance use train-compatible
ring profiles; pupil radius varies over 1.2..3mm at .3mm steps. Every compatible
assignment is retained in a factored family. Selection uses training boundary
residuals only; it is not a continuous trajectory or temporal smoothness fit.

| Diagnostic | Left | Right |
|---|---:|---:|
| Per-frame optical rows tested | 205,821 | 210,924 |
| Shared tuples compatible at 2px | 34/81 | 57/81 |
| Compatible per-frame assignments | 16,278 | 46,605 |
| Shared tuples at 1px / 3px pupil tolerance | 4 / 71 | 26 / 81 |
| Shared tuples with centred pupil | 1 | 9 |
| Shared tuples with depth fixed at 3.6mm | 11 | 18 |

Selected shared tuple (radius, depth, dx, dy): left (8.4,3.9,-.6,.6)mm; right
(7.2,3.9,-.6,0)mm. Several parameters are at grid limits. Every observation's
family still reaches -40 and +40 degrees yaw. A narrow sampled subset is not
confidence. Added pupil/depth freedom and a denser pupil-size grid remove the
previous empty-family case, but their individual contributions are confounded.

All six new held limbus and four held pupil points are within 2px. Joint held
limbus distances are .125941/.132415/.408211 left and
.865320/1.157500/.024454 right; frontal controls are
.505054/.855249/.355277 and .715669/.959153/.156489. Held pupil distances are
.751946/.436356 left and .096953/.056153 right. These sparse checks do not
identify hidden anatomy or material coordinates.

Exploratory controls AFTER frozen primary scoring shift training pupil marks
by one pixel (x-/x+/y-/y+). Shared tuple counts become 12/49/41/24 left and
40/71/72/44 right. A midpoint pupil-size grid (1.35..2.85mm) leaves 34 left/59
right and changes the selected right corneal radius from 7.2 to 8.4mm. Nothing
was retuned or substituted into the primary result using those controls.

## 3. Same-eye observed-RGB transport and limits

The best boundary-conditioned hypothesis maps target pixels through the refracted
iris plane into four same-eye donors. It assumes rigid iris coordinates and zero
relative torsion. Target BOX samples and all four bilinear donor pixels with
their 3x3 BOX samples must be candidate iris and conservatively within annotated
lids/outside reflection polygons. Margin: 2px plus sqrt(2). No target RGB is
used for selecting coordinates, donors or support. A frontal centre/scale control
uses identical paired scoring support. Equal donor means are formed in linear RGB.

Median per-pixel display RGB absolute error, in 0..255 levels:

| Three-or-more donor mean | Pixels | Joint | Centre/scale |
|---|---:|---:|---:|
| Left | 117 | 5.819331 | 5.561056 |
| Right | 110 | 5.412190 | 5.948307 |

Mixed results, not an established correspondence improvement. Mean prediction
abstains on 254/371 eligible left pixels and 256/366 right. These are photographed
RGB sums at candidate iris locations, not certified clean-radiance accuracy.
Original reflection polygons are excluded, not reconstructed. No pupil is forced
black. The check cannot establish temporal material stability or removal.

Even best assignments at each shared tuple give geometry-only median lower
bounds on donor-coordinate diameter of **3.165395px left / 2.955371px right**.
462/493 left and 462/471 right paired pixel/donor cases exceed 2px. These are
centre-ray geometry witnesses without full visibility support at every witness;
they are ambiguity evidence, not extra usable donors. Added +/-10-degree texture
orientation witnesses test unmeasured torsion. Neither subset enumerates the
full factored family, so agreement would not certify it. Lid uncertainty changes
support; no score or route is promoted into a clean/reliable removal mask.

Sharp limbus, single effective interface, assumed scale/camera, iris rigidity and
incomplete reflection exclusions remain limitations. Iris-depth sensitivity is
not posterior-cornea/lens modelling. Smooth anatomy, pupil deformation, true
camera PSF, credible 3D lid/light occlusion, source-material/light decomposition
and production scene registration remain unresolved. No real scene reflection
term was computed for these fits; the coupled transaction stays disabled.

## 4. Verification and review

Initial inverse transport started at the unrefracted iris projection. Signed
controls caught six failures at oblique gaze, maximum coordinate error 3.035406px.
They were flagged invalid; no production change occurred. Starting at the visible
apex with descending Newton steps recovers all **2,513 rays in seven signed and
asymmetric-camera cases**, maximum roundtrip 2.232696e-9px. Initial predictions
and report are preserved in `composite/r26_joint/initial_inverse`; corrected
real-source paired support is identical and primary scores effectively unchanged.
See `logs/r26_inverse_correction.json`.

Full replay of all 416,745 optical rows, families, mappings, predictions, witnesses
and scores passes. Held-pupil substitution and check-column replacement do not
alter conditioning; held-arc substitution preserves nine train-only profiles per
eye. Two target-RGB replacement read guards per eye confirm zero target reads
inside mapping/prediction and identical predictions. Original archives are never
modified. Decentration sign, complete interpolation footprints, empty family,
closure and independent affine-image interpolation controls pass.

**300 exact keyed-eye fallbacks, 304 protected hashes and ten new disabled complete
transactions pass.** Full audit, verifier and three encodes exit zero.

- `final/r26_joint.mp4`: 150 continuous frames, 60000/1001fps, 768x960, 2.5025s.
  Source left/diagnostic right, separate eyes; overlays at five observations,
  explicit abstention elsewhere and through blinks. Seven decoded contact sheets
  and the native decoded target board inspected. Max mean encode error 1.892521.
- `final/r26_detail.mp4`: 120 frames, 24fps, 768x992, 5s. Five labelled observation
  holds, not motion. All five distinct decoded holds inspected. Max error 1.494803.
- `final/r26_transport.mp4`: 120 frames, 24fps, 768x722, 5s. Four individual donor
  checks and the mean, each held one second. Same support in source/joint/control
  panels; black means unscored. All five distinct decoded holds inspected. Small
  lower/side iris regions show remaining RGB differences. Max encode error .070712.

Reviews are display-sRGB/CRF12, not HDR masters or artist acceptance. Original
reflections remain. No live playback. Exact-source claims are on linear arrays,
not lossy video. Reports are `logs/r26_*`, artifacts under `composite/r26_*`.
Existing AGENTS/round-17 work and the generic latest review log are excluded
from this round's commit.

```powershell
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/prepare_r26_source.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r26_joint.py --shot music_videos/reporter_test/shots/reporter_a --freeze-only
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r26_joint.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/review_r26_transport.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r26_joint/review --fps shot --out r26_joint.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r26_detail/review --fps 24 --out r26_detail.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r26_transport/review --fps 24 --out r26_transport.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r26_joint.py --shot music_videos/reporter_test/shots/reporter_a
```

## 5. Starter prompt for round 27

> Stay on eyes. Read round26.md and docs/r26-eye-findings.md. Five source
> observations per eye now share optical shape/decentration candidates, with
> complete factored ambiguity retained. Boundary checks pass, but same-eye
> observed-RGB prediction is mixed and compatible geometry alone shifts material
> correspondence by several pixels. Source 805 and all round-26 scores are now
> seen; preserve a fresh check in any extension. Move toward a joint source
> material/geometry/light fit that can use observed photometry while holding out
> target RGB and retaining competing explanations for moving reflections. Do not
> repeat a boundary-only fit or promote donor-copy predictions/route dots into
> clean material. Model/grid and one-pixel label sensitivity remain substantial;
> pupil deformation, posterior optics, smooth limbus and 3D lid/light visibility
> are unresolved. A narrow finite family is not confidence. Keep complete old
> reflection removal plus actual scene-light replacement in scope, with exact
> keyed-source fallback until the entire transaction earns support. No generated
> performer texture, key retuning, capture requirements or impossibility claim.
> Produce a temporal review and round27.md with measured findings and limits.
