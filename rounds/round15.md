# Round 15 handoff — RAFT improves observed correspondence, removal still unsupported

**Date:** 2026-09-08. Starts from round 14; production defaults remain unchanged.
**Delivered:** reusable pretrained dense correspondence adapter, frozen nearby-donor
comparison, hidden-RGB and native-sampling verification, blink diagnostics, review video.
**Not delivered:** independently clean donor masks, reliable material visibility,
complete original-reflection removal, or scene-light reflection replacement. Original
catchlights, crown sheen and lip gloss remain unresolved. The photorealism bar is unmet.

## 1. Artist decisions and scope — preserve these

- Generic prepared footage is the product. This reporter-only audit supplies data
  to a reusable array API; it does not establish generalization to another clip.
- Preserve photographed performer appearance. Neural predictions supply coordinates
  only; sampled RGB comes from the same eye's observed source frames.
- Keep exact keyed source, including its original reflection, when evidence fails.
  Only permit replacement together with independently validated complete removal.
- Tracking, removal and actual scene-light replacement remain the full task.
  Corneal pose, lid/lash/brow visibility and incident-light occlusion must be
  validated before enabling the Cycles reflection composite.
- No new capture reference requirement, generated performer texture, key retuning,
  gate relaxation, or reinterpretation of a reporter failure as universal impossibility.

## 2. Delivered implementation and evaluation separation

`scripts/dense_iris_flow.py` wraps **torchvision 0.23 RAFT large C_T_SKHT_V2** in the
existing `dependencies/ellseg-runtime/.venv` (torch 2.8/cu128). No packages or prior
checkpoints were upgraded. The official 20.1 MiB weights are in
`dependencies/raft-weights/raft_large_C_T_SKHT_V2-ff5fadd5.pth`; full SHA256:
`ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322`.
Download and cached loads validate the published hash prefix.

The [official model documentation](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.optical_flow.raft_large.html)
and [inference example](https://docs.pytorch.org/vision/0.23/auto_examples/others/plot_optical_flow.html)
establish the released weights and preprocessing. The adapter runs 12 updates in
float32 on CUDA, scrubs exclusions before bilinear resizing and normalization,
honors the supplied pixel aspect, and returns native displacement. Both resized
dimensions are at least 128 and divisible by eight. The reporter uses square pixels;
it never inherits the artist's camera rig. RGB sampling uses explicit native
bilinear coefficients, avoiding OpenCV's fractional-coordinate table rounding.

`scripts/audit_r15_dense.py --shot` is explicitly reporter-only. The reusable module
has no reporter coordinates or labels. This round **reuses** the 30 round 14 iris
points and 12 sparse reflection points in 959/977/995, both eyes. They remain seen
assistant annotations with 1.5-pixel boundary uncertainty, not independent human
truth, dense clean-material masks, or newly unseen footage.

`docs/r15-evaluation-protocol.md` was written before RAFT inference. Each target has
six fixed same-eye donors at offsets -12, -8, -4, +4, +8, +12. There is no ranking,
torsion search or per-case fitting. This nearby pool differs from round 14's distant
pool; **do not rank the two rounds numerically**. The new comparison is RAFT versus
zero displacement in the same tracked crops with identical donors and target points.

All five known iris pixels in each eye image are hidden simultaneously. Both flow
directions and target EllSeg/RITnet proposals are recomputed from scrubbed images.
All query and wrong-query footprints across both maps are excluded from every
donor context/control sample. This protects photometric scoring from shared native
contributors. **Context is not held out from the frozen neural network's receptive
field:** RAFT sees surrounding target and donor RGB. It has no local optimization
whose training/validation split could be claimed. The fixed face tracker/crop seed
also remains outside the hidden-RGB guarantee. Existing round 14 search isolation
is retained and tested without changing that path.

## 3. Real findings — useful mapping signal, insufficient material evidence

All counts below retain the full 30 attempted known-iris pixels. Support requires
three donors; candidate support additionally requires all four interpolation
contributors to lie in the eroded combined donor iris proposal. That proposal is
still not an independently validated clean-material mask.

| Metric | Fixed tracked crops | RAFT |
|---|---:|---:|
| Raw in-bounds supported pixels / 30 | 30 | 30 |
| Median per-pixel display MAE, paired 30 | 5.8333 | 2.6596 |
| p95 per-pixel display MAE, paired 30 | 31.2500 | 11.8145 |
| Raw donor MAD <=2 pixels / 30 | 0 | 11 |
| Candidate-supported pixels / 30 | 6 | 13 |
| Candidate support AND donor MAD <=2 / 30 | 1 | 9 |
| Paired candidate-supported pixels | 4 | 4 |
| Median per-pixel display MAE on those four | 13.3333 | 2.5511 |
| p95 per-pixel display MAE on those four | 15.9333 | 7.3205 |
| Pixels passing complete experimental gates | **0** | **0** |

These are absolute errors in source display RGB code values, not scene-linear
relighting error or certified clean-iris accuracy. Raw support allows any in-bounds
donor material; it is useful for a coordinate baseline, not safe replacement.

**The context test is underpowered, not evidence that all RAFT correspondences are
wrong.** After masked neural inference and all-arm native exclusions, each of the
36 iris donor pairs has **0–3 common context pixels**. None reaches the fixed
minimum of eight. Five of six masked target eyes also fail candidate area/pupil
gates; only 995 right passes. Thus no context gate is evaluable, and no gated
reconstruction is supported. Do not lower thresholds or call zero repairs proof
that the measured mapping improvement is meaningless.

**All six reflection probes / 12 marked reflection pixels have zero candidate
donor support**, for both methods, despite six in-bounds samples per point. No
hidden clean truth exists at these points, so their bright source RGB is never
scored as clean reconstruction truth. Full reflection extent and pupil crossing
remain unknown. Better lower-iris mapping does not solve these missing observations.

**Blink stress:** direct maps from the five labelled points at 977 to every frame
950–999 show why round-trip consistency is not visibility. At nearly closed source
981 left, **4/5 points pass <=0.5 native px** while no mapped footprint passes the
candidate material mask. At 982 left, 1/5 passes. At the inherited false candidate
984 left, the neural eye gate still accepts (51 iris candidates, 27 pupil pixels),
but 0/5 points pass the cycle test. At 985 both eyes, 0/5 pass. This is diagnostic
evidence on reused blink failures, not a validated visibility classifier. Full
boards show tracks continuing onto the closed-lid region. Green overlay points
mean only cycle plus candidate support; they do not authorize repair.

## 4. Verification and review artifacts

`scripts/verify_r15.py --shot` passes:

- Analytic bilinear radiance sampling: maximum error **0.0**. Known displacement
  mapping at four directions and square/anamorphic pixel aspects: max error **0.0**.
- RAFT inference on eight known translations of an observed eye image: median
  native endpoint errors **0.00984–0.03545 px**, worst p95 **0.04836 px**. This checks
  direction/scaling on controlled translations, not real iris deformation quality.
- All **12 masked cases** (six iris, six reflection), with both black and magenta
  substitutions, give exact flow fields in both directions, proposals, samples,
  counts, gates and metadata. **24 substitution recomputations** in total.
- All **72 saved real donor pairs** independently re-sample original donor RGB from
  recorded flow coordinates and satisfy all-arm context/query footprint exclusions.
  A synthetic poison test verifies query changes cannot affect disjoint scoring.
- Round 14's full-search poison regression still passes: 223 searched footprints,
  428 unchanged training pixels, unchanged selected angle, changed validation error
  144.9259. It concerns the old fitting path, not RAFT's neural receptive field.
- All **300 keyed-eye crops** remain exactly equal to source frame regions; fallback
  changes zero channels. All **304 protected hashes** match.
- New scripts compile. Both command-line audit/verifier list shots without `--shot`.
- Review fully decodes: **50 frames, 60000/1001 fps, CRF 12**, 960x736, 1,556,651 bytes.

Review: `music_videos/reporter_test/shots/reporter_a/final/r15_dense.mp4`.
Left is source, middle is RAFT point diagnostics, right is exact source fallback.
It is **0.834 seconds of source motion, not a reflection-fix video**. Video panels
are source-space crops; exact keyed-space fallback is verified separately. All
50 ordered decoded thumbnails and full boards 980/981/982/984/985 were inspected.
Original softboxes remain visible on the source/fallback sides when the eyes open.
This was contact-sheet and frame review, not real-time playback or artist approval.

Evidence: `logs/r15_dense.json`, `logs/r15_verification.json`,
`logs/r15_visual_review.json` under the shot. Flow NPZs and PNGs are reproducible
ignored caches under `composite/r15_dense/`. The verifier records code hashes,
checkpoint hash and review hash; visual review has a separate record so rerunning
verification cannot silently imply visual approval.

```powershell
& dependencies/ellseg-runtime/.venv/Scripts/python.exe scripts/audit_r15_dense.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r15_dense/review --fps shot --out r15_dense.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe scripts/verify_r15.py --shot music_videos/reporter_test/shots/reporter_a
```

Use `--review-only` on the audit to redraw the cached diagnostics. No production
renderer/compositor, scene, original footage or prior neural checkpoint changed.

## 5. Remaining work and next decision

Preserve RAFT as a measured nearby correspondence baseline. The next useful test
should independently label **visible material and donor footprint cleanliness**,
including genuinely occluded negatives, before claiming reliability. Sparse
target labels and eroded segmentation proposals leave a validation bottleneck.
Declare new versus reused samples, and preregister a test with enough observable
context rather than retrospectively weakening this round's gate.

A temporal visibility model such as the authors' released
[CoTracker3](https://github.com/facebookresearch/co-tracker) can be compared with
RAFT plus segmentation on those negatives. **It has not been run here.** Visibility
predictions still need evaluation on the iris, pupil, lid and reflection boundary;
stable tracks through a closed eye are not clean donor evidence. This does not
rule out reliable reconstruction on other prepared footage.

Complete source-removal coverage must precede coupled scene replacement. Keep the
Cycles work explicit: solve real-shot corneal pose and pupil crossing, replace the
flat composite cap with validated visible corneal coverage, and represent lid/lash/
brow occlusion for both camera and incident-light paths. Code inspection confirms
`stress_proxy.py` still positions corneal spheres and relies on the smooth proxy
occluder; this round made **no new render or occlusion-validation claim**. Hair sheen
and lip gloss received no treatment and need their own observed-material separation.

## 6. Starter prompt for round 16

> Read round15.md and the frozen protocol. RAFT improves nearby-donor error on all
> 30 reused hidden iris pixels (median 5.8333 to 2.6596) but proves no clean repair.
> All 12 reflection probes have zero candidate donor support. The context gate is
> unevaluable (at most three pixels against eight required), not evidence of failed
> RAFT correspondence everywhere. Keep this baseline; do not relax gates or repeat
> an underpowered test as progress. Design independent visible-material/donor
> validation with occluded negatives, and evaluate a released temporal visibility
> component such as CoTracker3. At closed-eye source 981 left, 4/5 RAFT cycles pass;
> low cycle error is not visibility. Declare new/reused/post-hoc evidence, preserve
> hidden-RGB and native-footprint controls, and keep exact keyed fallback. Tracking,
> complete source removal and actual scene-light replacement remain the task;
> corneal pose, pupil crossing and camera/incident lid-lash-brow occlusion remain
> prerequisites. Produce an honestly labelled temporal review and write round16.md
> with measured remaining blockers. No generated performer texture or forced repair.
