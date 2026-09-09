# Round 17 handoff — reflection tracks reach reflections, clean donor identity unproven

**Date:** 2026-09-09. Starts from round 16; production defaults unchanged.
**Delivered:** RGB-free single-query context preflight, all 12 real reflection
queries through both retained pretrained trackers, independent material/sampling
verification, and a labelled temporal review.
**Not delivered:** clean iris correspondence under reflections, reflection
removal or scene-light replacement. Original catchlights, crown sheen and lip
gloss remain unresolved. Photorealism is unmet.

## 1. Artist decisions and scope — preserve these

- Generic prepared footage is the product; reporter results are bounded evidence.
- Reconstruct from observed RGB of the same iris only where identity, material
  and visibility are reliable. Models supply coordinates, never performer RGB.
- Exact source-keyed fallback, including its original catchlight, is accepted.
- Tracking plus scene-light replacement remains the full task. A visibility
  result cannot authorize source removal or added scene reflection.
- Full reflection extent, pupil crossing, real corneal pose and camera/incident
  lid-lash-brow occlusion still precede enabling the Cycles reflection composite.
- No key retuning, generated texture, weaker gates or required capture references.

## 2. Frozen evaluation and implementation

`docs/r17-evaluation-protocol.md` was written before new inference or RGB scoring.
It freezes two separate experiments: context capacity without RGB, and material
classification at reflection-anchored tracks. Neither is a clean-radiance test.

`scripts/context_capacity.py` is a reusable array API accepting native material
masks and one query. It enumerates complete 2x2 context footprints, removes the
target query footprint, and excludes donor contributors shared with the fixed
query or its +3-native-pixel horizontal wrong-map control. Both fixed and wrong
context maps must fit inside the same frozen donor iris interior. Adding the
RAFT/CoTracker maps and their query exclusions can only reduce that intersection.
The returned capacity is therefore an upper bound **for this control design**,
not a bound on recoverability by every possible method.

`scripts/audit_r17_reflection.py --shot` is reporter-only and uses the generic
capacity API plus the existing unmodified pretrained adapters. It requires a
saved `--preflight-only` result before inference. `--review-only` redraws caches.

New `scripts/data/r17_context_labels.json` contains six conservative iris
polygons on source 959/977/995, both eyes. Existing source-only native grids were
inspected before writing these labels and before this round's predictions.
These are assistant annotations on previously seen footage, not an independent
annotator or clean-material truth. All round 16 labels, including unknowns,
remain byte-identical. The same 1.5 px boundary uncertainty is used throughout.

**Preflight:** all 30 round 14 iris queries and 12 reflection probes, one hidden
query at a time. Fixed same-eye donors are 974, 990, 994, spaced by at least four
source frames. Target context comes from the new polygons; donor context comes
from unchanged round 16 iris regions. No RGB error, donor ranking or parameter
search is consulted.

**Reflection material:** all 12 round 14 reflection points, separately, anchored
at 959/977/995. CoTracker sees the same 50 native eye crops, source 950..999.
RAFT directly maps each target to the eight unchanged round 16 labelled frames:
974, 978, 981, 982, 984, 985, 990, 994. This gives **96 attempts per arm**.
These are different donors and exclusion footprints from round 15; do not rank
their numbers as a paired improvement over round 15.

Each target's complete 2x2 native query footprint is scrubbed before resize,
including both RAFT directions. Each query runs alone. Surrounding reflection
RGB remains visible: this is NOT complete reflection masking, and a network
can still follow the softbox feature. Fixed upstream crop/landmark seeds remain
outside hidden-RGB guarantees. CoTracker forces query-frame output; none of the
eight scored donor frames is a query frame.

Released CoTracker3 scaled offline and RAFT large C_T_SKHT_V2 use exactly the
round 16 and round 15 settings, weights and adapters. No package, model checkout
or checkpoint changed. Model provenance matches the previous reports exactly.

## 3. Results — no donor repair earned

### Context capacity, before inference

- **42 queries, 126 target/donor pairs.** Maximum common context capacity: eight.
- Only **5/126 pairs** can even supply eight samples under the fixed controls.
- **0/42 queries** have three capacity-eligible donors. All are insufficient.
- **Zero photometric scores computed.** No inferred context fit, MAD result or
  withheld-pixel accuracy is claimed.

Hiding one query at a time does not by itself make this test adequately powered.
The frozen iris interiors and the fixed/+3px control intersection leave too
little common material across these donors. This demonstrates a limitation of
these annotations and controls, not universal impossibility on this clip or on
other footage. Do not compare these counts directly with round 15's different
neural masks, donor pool and five-point exclusion.

The thresholds are unchanged: >=8 identical context samples per donor,
MAE <=3 display levels, wrong-map margin >=1, >=3 spaced donors, MAD <=2;
RAFT query cycle <=0.5 native px. None was evaluated as a complete repair gate.

### Actual reflection-region donor material

Material columns classify **accepted** complete native footprints only.
Unknown includes unlabelled and boundary footprints; it never means clean.

| Arm | Attempts | Accepted | Iris-compatible | Labelled reflection | Unknown |
|---|---:|---:|---:|---:|---:|
| RAFT cycle <=0.5 px + native bounds | 96 | 27 | 0 | 14 | 13 |
| RAFT cycle + same neural iris proposal | 96 | 0 | 0 | 0 | 0 |
| CoTracker visibility + native bounds | 96 | 15 | 0 | 8 | 7 |
| CoTracker visibility + same neural iris proposal | 96 | 0 | 0 | 0 | 0 |

No accepted footprint belongs to the other explicit material classes. Both base
arms reject all **24 closure attempts** at 981/982; these are correlated points
from one blink, not 24 independent events. No query has three iris-compatible
donors. There is no ground-truth clean RGB beneath these source reflections, so
the original highlight RGB is never scored as the desired answer.

The practical finding is that passing tracking/cycle tests at reflection
anchors can still put samples in **another photographed reflection**. This
directly exercises the desired repair region, unlike the lower-iris visibility
queries in round 16. It does not establish the material motion beneath it.
Even a future iris-compatible footprint would still need same-point identity
and clean-radiance evidence.

## 4. Verification

`scripts/verify_r17.py --shot` completed with exit 0 and checks:

- Independent contributor-by-contributor enumeration of all **126 context
  capacity pairs**, including target hiding and fixed/wrong donor exclusions.
- A deliberate donor-query RGB corruption leaves retained context samples
  exactly unchanged for both control maps; the query samples themselves change.
  An extra arm's query exclusion only removes capacity.
- **696 native coordinates** independently resampled: 600 CoTracker temporal
  coordinates and 96 RAFT donor coordinates. Maximum CoTracker sampling
  difference **0.0**; every saved RAFT donor RGB equals the independent sampler
  exactly. Complete material footprints independently reclassified for all arms.
- **24 full inference recomputations**: black and magenta substitutions on all
  four excluded contributors for each of 12 queries. **528 array-hash
  comparisons** match exactly, covering both RAFT fields, all CoTracker outputs,
  exclusions and query inputs. Material outcomes are recomputed from the saved
  coordinates and unchanged masks.
- All **100 source crop hashes**, **300 keyed-eye fallbacks** and **304 protected
  files** unchanged. Seven prior adapter/audit/verifier code hashes match.
- Round 14 search-isolation regression passes: 223 searched footprints,
  428 unchanged training pixels, unchanged selected angle; deliberately changed
  validation MAE 144.9259. This remains a controlled regression, not real accuracy.
- Protocol, label, proposal, model and code provenance checked; scripts compile;
  no-`--shot` commands list available shots; `git diff --check` passes.

The initial redirected audit shell returned exit 1 from the already-known
PowerShell treatment of git's harmless global-ignore permission warning as an
error record. The log contains all 12 completed queries and the final report,
with no Python traceback. The independent verifier exited 0. Review-only and
encoding also exited 0. See `logs/r17_run.log`; do not cite the shell error as
an inference failure or its final report alone as verification.

## 5. Review and reproduction

Review: `music_videos/reporter_test/shots/reporter_a/final/r17_reflection.mp4`.
**50 frames, 60000/1001 fps, 1280x736, CRF 12**, about 0.834 seconds. Panels:
source / masked-query RAFT / masked-query CoTracker / exact source fallback.
The four displayed reflection queries are anchored at 977; the numerical audit
covers all 12 queries. RAFT is drawn only at the eight evaluated donor frames;
other frames explicitly say not evaluated. No interpolated RAFT track is shown.
CoTracker's query frame is labelled forced. Green means cycle or visibility,
never clean iris. Source/fallback retain the original reflections.

All 50 decoded thumbnails were inspected in order, plus full boards at 974,
977, 981, 982, 984, 985, 990, 994. At 990, RAFT's 977-anchored queries pass cycle
while CoTracker keeps them rejected; markers still sit near the softbox.
This is a post-inference visual observation, not a new label or accuracy score.
Review is contact-sheet/full-frame inspection, not real-time playback or artist
approval. `logs/r17_visual_review.json` records the inspection separately.

```powershell
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r17_reflection.py --shot music_videos/reporter_test/shots/reporter_a --preflight-only
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r17_reflection.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r17_reflection/review --fps shot --out r17_reflection.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r17.py --shot music_videos/reporter_test/shots/reporter_a
```

Reports: `logs/r17_preflight.json`, `logs/r17_reflection.json`,
`logs/r17_verification.json`. Reproducible NPZ fields/tracks, review PNGs and
decoded sheets: `composite/r17_reflection/`. `logs/review_video.json` describes
this review. Previous source/model caches are required; production renderer,
compositor, scenes and original footage were not changed.

## 6. Remaining work and starter prompt for round 18

The next missing fact is **same material point identity beneath a moving
reflection**. Anchoring on a softbox and observing another softbox is not that
fact. A useful next controlled test should separate material motion from
reflection motion, using observed iris RGB with known transforms and explicitly
synthetic additive reflections. Compare reflection anchors against observed
iris anchors tracked toward contaminated targets; assess whether a model locks
onto the reflection. Keep the result explicitly synthetic. Do not turn a
successful geometric visibility test into a real clean-radiance claim.

Before another real photometric comparison, preflight a new design whose
controls fit adequately observable material. Changing controls must be frozen
and validated for discrimination, not selected by favourable RGB errors. Do
not weaken the eight-pixel gate or expand unknown labels after seeing errors.
A different prepared clip with resolvable iris texture remains valid; this
reporter result does not rule out temporal reconstruction generally.

> Read round17.md and the frozen protocols. All 12 real reflection queries now
> run singly through RAFT and CoTracker. Their accepted labelled donor footprints
> are reflections or unknowns, never iris-compatible; the common neural proposal
> rejects all 96 attempts in both arms. The fixed/+3px source-label capacity
> preflight has zero of 42 queries with three eligible donors, so no photometric
> scoring was done. Move toward same-point material identity rather than another
> visibility count: distinguish iris motion from reflection motion with known
> geometry and observed texture, then preflight an adequately powered real
> design before scoring. Preserve all-arm native exclusions, hidden RGB,
> unchanged gates, exact source fallback and coupled removal/replacement.
> Full clean coverage, pupil crossing, real corneal pose and camera/incident
> lid-lash-brow occlusion still block scene reflections. Hair/lip separation
> remains open. Produce a labelled temporal review and write round18.md;
> no generated performer texture or forced repair.
