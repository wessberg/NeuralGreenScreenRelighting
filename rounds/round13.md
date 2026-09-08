# Round 13 handoff — pretrained eye models executed, visibility still unvalidated

**Date:** 2026-09-08. Starts from round 12; production defaults remain unchanged.
**Delivered:** runnable pretrained EllSeg/RITnet adapters, frozen real-crop
evaluation, reusable keyed-eye observations and three review videos.
**Not delivered:** reliable reflection removal, correspondence, occlusion-correct
scene-reflection replacement, or a photorealism acceptance claim. The original
catchlights, crown sheen and lip gloss remain unresolved.

## 1. Artist decisions and scope — preserve these

- The tool must support arbitrary prepared footage. The reporter is one evaluation
  clip; generic APIs alone do not demonstrate generalization.
- Use released neural components before inventing more custom detection. This
  round actually downloaded and ran two author checkpoints, not just their APIs.
- Tracking, removal **and scene-light-based replacement** remain the full task.
  Neither segmentation model removes reflections or solves incident-light occlusion.
- Unreliable eyes keep exact keyed source and the original catchlight. Removal and
  replacement must be coupled per eye; never stack a synthetic reflection onto a
  failed source-removal result. Use the existing corneal Cycles renderer.
- No generated performer texture, new on-set references, or reporter key retuning.

## 2. Delivered code and isolated runtime

The new third environment is `dependencies/ellseg-runtime/.venv`: Python 3.13.14,
torch 2.8.0+cu128, torchvision .23+cu128, NumPy 2.4.2, OpenCV 4.13. Both models
executed on the RTX 5080. Pipeline/facetrack packages were not modified.

Author sources and checkpoint hashes are recorded in
[docs/neural-eye-evaluation.md](../docs/neural-eye-evaluation.md), with pinned
commits, licenses, setup commands and `scripts/data/r13-neural-runtime.txt`.
Checkouts and weights are ignored under `dependencies/`; no weights enter Git.

- `ellseg_iris.py`: strict-loads EllSeg's `weights/all.git_ok`, runs the original
  DenseElNet encoder/decoder/regressor and maps complete ellipses to native pixels.
  In-memory `np.int` scalar-cast compatibility is the only architecture substitution.
  Integer crop/pad fixes the author's Python-3 slice bug. Actual half-pixel mapping
  and rounded dimensions are recorded, including pixel aspect. Unsampled pixels
  are unknown, and constant images abstain. Original checkout stays unmodified.
- `ritnet_iris.py`: strict-loads the original `best_model.pkl`, uses the author's
  grayscale/gamma/CLAHE/normalization and returns background/sclera/iris/pupil
  proposals. Aspect-preserving fit and zero pad into 400x640 is an **unvalidated
  input adaptation**, not proof of performance on general RGB footage.
- `neural_iris_observations.py --shot`: reuses dynamic native ROIs and track
  metadata. Stores both models' distinct proposals and original keyed RGB under
  `relight/neural_iris_observations/`, with origin, scale/roll and pixel mapping.
  Roll is metadata; neural crops are not roll-normalized. All geometry/visibility
  validation flags and `scene_reflection_allowed` remain false. Exact source
  fallback is exercised through the existing `apply_eye_repair` API.
- `audit_r13_ellseg.py`, `audit_r13_ritnet.py`: explicitly reporter-only adapters
  using round 10's source cache and round 12's unchanged sparse annotations.
- `verify_r13.py`: preprocessing/conic conventions, exclusions, fallback,
  protected hashes, full video decoding. Six new Python files compile.

The existing experimental native fitter, material sampler, correspondence tests,
ratio compositor, shot configuration, keys and corneal renderer are unchanged.
ProPainter, TSHRNet and other removal/propagation models were not run this round.

## 3. Real measurements — useful proposals, insufficient confidence

EllSeg evaluated **1,200 source frames / 2,400 eyes** (96.59 seconds including
diagnostic I/O). RITnet evaluated the selected 700–849 interval plus the five other
sparse-label frames: **155 source frames / 310 eyes**. Both additionally processed
**150 keyed frames / 300 dynamic eye crops** in the generic stage.

The source benchmark uses the same native 64x64 ROIs as round 12. The keyed stage
uses different crops and scene-linear-to-display conversion; do not mix results.
No crop/threshold tuning, fine tuning or training occurred on the sparse labels.
They are **reused assistant annotations**, not new held-out independent human data:
12 eye images, eight open and four rejected/closed; boundary uncertainty 1.5 pixels.

| Sparse labels | R12 candidate material | EllSeg complete iris | RITnet visible iris proposal |
|---|---:|---:|---:|
| Clean iris points admitted / 40 | 26 | 40 | 30 |
| Noniris points admitted / 60 | 0 | 16 | 0 |

The masks have different semantics, so this is **not a fair segmentation accuracy
ranking**. It directly demonstrates why EllSeg output cannot be a visible-material
mask. R12 masks are raw candidates, including rejected candidates. Sparse zero
noniris overlap does not establish dense visibility, much less clean donors.

EllSeg produces nonempty iris/pupil on all four closed samples. RITnet gives zero
iris/pupil in three, but source 79 left still has **10 iris and 4 pupil pixels**.
On source 923 left, RITnet misses substantial material and its pupil-boundary p95
error is **5.5912 native pixels**. No calibrated blink/visibility gate was derived.

For EllSeg versus R12 on the eight open images:

| Metric in native pixels | R12 | EllSeg |
|---|---:|---:|
| Median of per-eye pupil boundary median | 1.6472 | 1.0829 |
| Worst per-eye pupil boundary p95 | 5.3377 | 3.3405 |
| Median of per-eye limbus boundary median | 0.8587 | 1.2465 |
| Worst per-eye limbus boundary p95 | 7.0812 | 2.5288 |

Improvement is mixed, not universal. RITnet boundary metrics use raster contours;
EllSeg uses sampled continuous ellipses. Full scores and annotation provenance are
in `logs/r13_ellseg.json` and `logs/r13_ritnet.json`.

**Neither model distinguishes clean iris radiance from iris plus catchlight.**
RITnet's visible iris class can include the source reflection; EllSeg infers hidden
material. Neither can be promoted directly to a removal mask or clean donor map.
No positive same-eye repair evidence, reconstruction transition, temporal tracking
accuracy or additional-input generalization was established.

## 4. Verification and review artifacts

`verify_r13.py` passes:

- Original author commits/checkouts match. All checkpoint parameters strict-load.
- Upstream supported rectangular preprocessing max difference **2.3842e-7**.
- Author ellipse conic max residual **7.8044e-8**; extra sign checks use four angles.
- Two extreme hidden-RGB substitutions at each of two pixel aspects leave both
  models' outputs exactly invariant when scrubbed before preprocessing/inference.
  This uses **one real crop**, not a reconstruction test, and excludes the upstream
  face tracker. Existing hidden-footprint donor controls remain untouched.
- Constant input abstains. All **300 keyed eye crops** match source exactly and
  exercise fallback with **zero changed channels**.
- All **304 protected-file hashes** from round 10 match.
- All **1,500 video frames** decode at the shot's 60000/1001 fps.

CRF 12 review-only videos under the shot's `final/`:

- `r13_ellseg.mp4`: 1,200 source frames; source / R12 raw candidate / full ellipses.
- `r13_ritnet.mp4`: 150 source-motion frames; source / visible-part proposals.
- `r13_neural_observations.mp4`: 150 keyed frames; source / combined proposals /
  exact source fallback. **This is not a reflection-fix video.**

Visual inspection covered all six sparse RITnet boards, EllSeg source 79/779 boards,
and all 150 ordered thumbnails for each short video in aspect-preserving sheets.
The complete EllSeg sequence was decoded but not visually inspected. This was
image/contact-sheet review, not real-time playback or artist approval. Specific
observations are in `logs/r13_visual_review.json`; mechanical verification always
leaves visual review pending so a re-run cannot imply approval.

```powershell
& dependencies/ellseg-runtime/.venv/Scripts/python.exe scripts/audit_r13_ellseg.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe scripts/audit_r13_ritnet.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/ellseg-runtime/.venv/Scripts/python.exe scripts/neural_iris_observations.py --shot music_videos/reporter_test/shots/reporter_a --review
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r13_ellseg/review --fps shot --out r13_ellseg.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r13_ritnet/review --fps shot --out r13_ritnet.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r13_neural_observations --fps shot --out r13_neural_observations.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe scripts/verify_r13.py --shot music_videos/reporter_test/shots/reporter_a
```

## 5. Remaining work and next decision

Released neural components are now available here and have observed strengths and
failures. Keep their separate geometry/visibility semantics. RITnet is a useful
candidate for lid visibility, but its near-eye training domain, missing pupil regions
and partial-blink false positives still need calibrated abstention. EllSeg geometry
cannot establish visibility, corneal pose precision or iris-material correspondence.

Next establish whether neural proposals support independently checked clean
same-eye observations and correspondence. Preserve exclusions before all learned
operations and native interpolation-footprint controls. New validation must be
separated from any fitting/tuning; recorded sparse labels cannot become both.
The current iris sampler excludes pupil, so pupil-crossing source reflections still
need a separately validated path. Do not force their repair from absent observations.

The existing corneal rendering work remains part of the plan. Lid/lash/brow visibility,
incident-light occlusion and the previously rejected flat cap remain unresolved.
Reflection replacement stays coupled to validated complete source removal and exact
fallback. Hair sheen/lip gloss had no treatment this round.

## 6. Starter prompt for round 14

> Read round13.md and docs/neural-eye-evaluation.md. Released EllSeg and RITnet now
> run on the 5080 in an isolated environment; use these evaluated components rather
> than another custom contrast detector. The full task remains generic tracking,
> reliable same-eye source-reflection removal, and actual scene-light reflection
> replacement. Neural geometry and visible-part proposals are distinct, unvalidated,
> and include failures. Build and evaluate abstention/correspondence using fresh
> evaluation data separated from fitting; do not retune the reused sparse labels
> and call them held out. RITnet iris class is not clean material, and EllSeg's full
> ellipse is not visibility. Preserve hidden-RGB and native-footprint controls and
> exact source-keyed fallback. Test whether reliable observed same-eye samples can
> earn any real repairs, including explicit handling or abstention for pupil-crossing
> reflections. Keep renderer pose and lid/lash/brow/incident-source occlusion in the
> work plan; only enable its reflection where removal is validated. Any model-based
> removal benchmark must not silently turn generated texture into performer output.
> Produce a source-versus-result video if repairs are earned, otherwise clearly
> label diagnostic/fallback videos. Write round14.md with actual remaining blockers.
