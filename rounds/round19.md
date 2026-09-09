# Round 19 handoff — general reflection research and a released model deployed

**Date:** 2026-09-09. Starts from round 18. Production defaults unchanged.
**Delivered:** current-research correction, isolated UnReflectAnything deployment
and highlight-head evaluation, controlled local-motion experiment, registered
real-context capacity preflight, independent verification and two labelled videos.
**Not delivered:** clean performer reconstruction, source-reflection removal,
scene-reflection replacement, or instrument evaluation. Photorealism remains unmet.

## 1. Artist steering — this changes the next priority

- The artist challenged whether the latest research/models truly cannot reconstruct
  the iris. **We have not established that.** EllSeg/RITnet segmentation and
  RAFT/CoTracker tracking tests are not an exhaustive eye-reconstruction survey.
- A current source check found a concrete earlier mistake: UnReflectAnything's
  GitHub README says weights are unreleased, but the author's Hugging Face files
  and current loader contradict that. The full checkpoint now runs here.
- The artist explicitly includes **playing drums and guitars**, with reflective
  instruments and hardware. A general material-separation strategy is desired;
  do not spend another round only extending reporter-eye tracking benchmarks.
- The artist questioned the bundled SCARED samples. They are endoscopic surgical
  images. One was used for an integration check; no reconstructed RGB was saved
  or shown and no surgical imagery appears in the reviews. Use ordinary scene or
  object images for further checks. The verifier no longer automatically runs
  the bundled surgical sample; new checks require an explicit image path.
- Preserve same-performer observations, identity, input-agnostic design, unchanged
  key thresholds/gates and exact keyed-source fallback. Source fallback including
  original catchlights is accepted. No capture reference is a requirement.
- Tracking plus coupled removal/scene-light replacement remains the full task.

## 2. Current research and UnReflectAnything

Read `docs/r19-material-research.md` for cited primary sources, deployment and the
proposed general-material architecture. The 2026 *Reconstructing Realistic and
Relightable Eyes* paper uses light-stage captures; checked author/paper pages did
not expose code/weights. The open *Seeing the World through Your Eyes* jointly
fits iris texture, corneal pose and reflected scene but remains unrun here.
These are different from the trackers we tested, and from each other. Do not
claim impossibility, an exhaustive latest-method search, or a universal solution.

UnReflectAnything author code: `363ebb5c87d9f4f49b63f744fded8d3786a051b3`.
HF full checkpoint revision: `ed889890c625698c8665d872fc02987bafd1b134`.
Checkpoint SHA256: `12c0cf4518a39325b7bfe2bed04a96395189f2643c2f90b7b79360a659dc739b`.
Encoder mirror revision: `f43714afb20addc93acad542bf6ea0376b6f7900`.

Separate runtime: `dependencies/unreflect-runtime/.venv/Scripts/python.exe`.
Author-pinned torch 2.9.1/cu128, torchvision 0.24.1, Transformers commit
`2fe43376cdde02b7ffcf117e6eb9aa4375fb2dd1`, numpy 2.4.2, OpenCV 4.13.0.
Pipeline, facetrack and earlier neural environments were not changed.

`unreflect_highlights.py` exposes an arbitrary observed-RGB array API, loading
with weights_only=True and an explicit DotMap allowlist. All **601 state entries
match exactly** after the author's checkpoint-prefix normalization. It uses the
embedded training config: token-inpainting local-prior weight **0.2**, whereas
the packaged YAML says 0.5. A strict weight load alone cannot catch this forward
parameter mismatch. No author architecture, weights or preprocessing was changed.

The supplementary `docs/r19-unreflect-protocol.md` froze the reporter test before
inference. Both native 64x64 eyes at source 950..999 pass through the released
896px preprocessing/encoder/highlight head. There is no performer token-inpainting
or diffuse-decoding output. Upsampling does not add observed iris information.

- **100 eye crops**, plus exact repeated inference on source 977 both eyes.
- **0/12 reflection probes**, **0/30 iris probes** above the author's 0.3 threshold.
  These are reused sparse annotations, not full clean-material masks.
- Only **three native pixels total** exceed 0.3 (952 left, 967 right, 976 right).
  None on source-labelled closure 981/982. Low response also occurs on open eyes;
  this does not validate visibility or meaningful blink rejection.
- Reflection probe probabilities: about **0.00073–0.04420**. No retuning performed.
- **28.986 seconds**, peak allocated CUDA **3,539,735,552 bytes**.
- Full-network integration on the initial bundled nonperformer sample returns
  finite diffuse output; its highlight map matches the extracted head exactly.
  That head reaches **0.95416** and has 16,313 model pixels above 0.3, so it is
  not universally silent. This is an integration check, not an instrument or iris
  reconstruction benchmark. The default verifier now preserves this old result.

The model is available and deployed. Its usefulness on this small-crop setup is
not demonstrated. Its full reconstruction capabilities on performer imagery and
ordinary reflective objects remain **untested**, not disproved. A detector-only
failure must not become a claim that the diffuse reconstruction model cannot work.

## 3. Frozen local material motion test

`docs/r19-evaluation-protocol.md` predates all new inference/capacity counts.
`local_reflection_motion.py` moves observed texture inside a stationary aperture:
exact translated pixels within radius 8, exact static surroundings beyond radius
12, and a linear-light raised-cosine blend between. Query footprints remain
inside the exact-motion area. This is an artificial local eye-texture patch,
**not a segmented iris, independently animated lids, or a refracting cornea**.

The eight round-18 point/axis definitions, 24 frames, two anchors, base/contaminated
conditions, reflection addition, released trackers and scoring rules are unchanged.
There are 32 single-query cases / 384 scored model observations. Each full case is
recomputed with black and magenta substitutions over its native query footprint.
20 clipped channel values occur across the eight unique synthetic sequences.
Original photographed highlights remain part of the observed base texture.

| Condition | Anchor | Model | Accepted / 48 | Median material EPE | p95 EPE | Accepted EPE >1 |
|---|---:|---|---:|---:|---:|---:|
| Base | 0 | RAFT | 28 | 0.9970 | 5.0068 | 8 |
| Added reflection | 0 | RAFT | 23 | 2.0071 | 4.9913 | 11 |
| Base | 12 | RAFT | 25 | 0.5878 | 2.0040 | 10 |
| Added reflection | 12 | RAFT | 30 | 0.9955 | 2.8463 | 18 |
| Base | 0 | CoTracker | 0 | 1.1759 | 4.0233 | 0 |
| Added reflection | 0 | CoTracker | 0 | 1.3088 | 4.2286 | 0 |
| Base | 12 | CoTracker | 5 | 0.3246 | 1.2996 | 0 |
| Added reflection | 12 | CoTracker | 2 | 4.7452 | 11.6331 | 2 |

EPE is native pixels. Both base and contaminated local-motion cases are harder
for RAFT than the hashed round-18 global-motion reference (all 48 accepted per
row, median EPE 0.0181–0.0254). No global reference inference was rerun.

Stationary-surround lock meets the frozen criterion on 6/1 RAFT base observations
for anchors 0/12, and 9/4 contaminated observations. CoTracker has none. Its two
accepted wrong contaminated predictions are one correlated `left_0_y`, anchor-12
sequence at t=0/4 (EPE 3.0444/3.1447).

**Three RAFT observations meet the reflection-distance criterion, but ALL THREE
also meet stationary lock.** They occur at t=16, anchor 0, in `left_3_x`,
`right_0_x`, `right_3_y`; the synthetic reflection is then one pixel from the
stationary anchor. These overlapping explanations prevent a causal reflection
lock-on claim. This ambiguity is reported, not hidden behind the criterion count.

The result demonstrates that global-crop motion was an easy case and cycle
consistency can accept wrong local tracks. It is not evidence against all neural
reconstruction, and there is no repair or clean-material truth.

## 4. Real RGB-free registered capacity

`registered_context.py` aligns frozen donor/target iris-mask centroids with an
integer translation, comparing it with the unchanged +3px horizontal control.
All 42 prior queries, 1.5px annotation uncertainty and donors 974/990/994 remain
fixed. Complete native contributors and both mapped-query footprints are excluded.
This replaces the old unregistered baseline; it is not the old all-arm intersection.

**9/126 pairs** have eight context samples (maximum eight), and **0/42 queries**
have three eligible spaced donors. Round 17 had 5/126 eligible pairs under its
different fixed-map design. Capacity is only an upper bound: learned-map/exclusion
intersections would reduce it. No RGB scores, control-discrimination validation
or repair were run. This design/label result is not a footage impossibility bound.

## 5. Verification, review and implementation bugs

- Local synthesis independently checked against OpenCV integer warps, scalar
  color/addition calculations, exact interior contributors and static exterior.
  Four signed directions pass. All 384 coordinates, cycles, bounds, decisions
  and summaries checked independently. Mapping max difference 0.00000190735 px;
  cycle max difference 0.00000337525 px.
- 64 complete inference substitution runs / 1,152 exact array-hash comparisons.
  Round-18 reference cache hashes and prior model/code hashes match.
- All **300 exact keyed-eye fallbacks** and **304 protected files** pass. Search
  isolation regression: 223 searched footprints, 428 unchanged training pixels,
  same selected angle; deliberately corrupted validation MAE 144.9259.
- Registered capacity independently enumerated for 126 pairs; four signed
  centroid controls verify direction. No relaxed MAE/MAD/context/donor gates.
- UnReflectAnything: 100 saved model/native maps hashed, native downsampling
  independently checked, 42 probe values and counts verified; installed upstream
  model/loader/inpainter files match the clean pinned checkout. Two exact repeats.
- Development failures fixed before completion: closed-eye annotations can omit
  the `iris` key; the new verifier initially imported EVAL from the wrong module
  and reused a scalar/function name. Loader equality checking initially forgot
  the upstream checkpoint-prefix normalization. No inference settings or gates
  changed in response to results. An attempted `.testruns` log path did not exist;
  actual run logs are in `logs/`. Final verifiers and encodes pass.

Reviews (display-referred sRGB, CRF 12, **no surgical images**):

- `final/r19_local.mp4`: **192 frames, 24fps, 1280x790, eight seconds**. Synthetic
  motion, base left, added reflection right, both models and anchors. Cyan material,
  yellow stationary anchor, magenta reflection, green/red model accept/reject.
- `final/r19_unreflect.mp4`: **50 frames, 60000/1001fps, 1280x400**. Original eyes
  left, raw soft probability maps right, labelled unvalidated/no reconstruction.

All 242 decoded thumbnails were inspected in ten ordered sheets; full decoded
local board 23 and UnReflectAnything board 31 were inspected, plus native review
board 27. Black probability panels are the measured weak response, not absent
frames. The local blend visibly deforms the eye patch relative to stationary lids;
it is an artificial test. No real-time playback or artist approval is claimed.
`logs/r19_visual_review.json` records this scope.

```powershell
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r19_local.py --shot music_videos/reporter_test/shots/reporter_a --preflight-only
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r19_local.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r19_local/review --fps 24 --out r19_local.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r19.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/unreflect-runtime/.venv/Scripts/python.exe -B scripts/audit_r19_unreflect.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r19_unreflect/review --fps shot --out r19_unreflect.mp4
& dependencies/unreflect-runtime/.venv/Scripts/python.exe -B scripts/verify_r19_unreflect.py --shot music_videos/reporter_test/shots/reporter_a
```

Reports are `logs/r19_local.json`, `r19_preflight.json`, `r19_verification.json`,
`r19_unreflect.json`, `r19_unreflect_verification.json`; raw caches/reviews under
`composite/r19_local/` and `composite/r19_unreflect/`. `--shot` omission lists shots.
The pre-existing uncommitted round-17 work and AGENTS edits were not modified.

## 6. Starter prompt for round 20

> Read round19.md and docs/r19-material-research.md. The artist wants a general
> reflection/material strategy for drums, guitars, hardware, skin, hair and eyes.
> The previous claim that UnReflectAnything weights were unavailable was wrong:
> the released full checkpoint now strictly loads all 601 entries in its isolated
> runtime, using the embedded 0.2 local-prior weight. Its highlight head runs but
> misses all 12 sparse reporter catchlight probes at threshold 0.3 on the frozen
> 64px crop setup. This is not a diffuse reconstruction test or a verdict on
> instruments. Prioritize a frozen ordinary-scene/object evaluation and inspect
> material reconstruction, geometry and scene-light replacement together. Do not
> automatically run/show surgical samples. Do not spend the next round merely
> repeating reporter tracker failures. Local independent-motion controls show
> accepted wrong RAFT tracks even without added reflections; the three apparent
> reflection locks are confounded with stationary-surround lock. Registered real
> context still has no query with three eligible donors, so no real RGB gate score
> is earned. Preserve observed performer identity, all-arm exclusions, exact source
> fallback and unchanged gates. No general impossibility claim, generated performer
> texture, key retuning or special capture references. Full reflection extent,
> corneal/pupil geometry and camera/incident occlusion still block eye replacement.
> Produce a labelled temporal review, measured claims and round20.md.
