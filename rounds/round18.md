# Round 18 handoff — material motion survives a small synthetic reflection in RAFT

**Date:** 2026-09-09. Starts from round 17; production defaults unchanged.
**Delivered:** frozen paired material/reflection motion experiment, reusable
observed-RGB synthesis API, both pretrained trackers in both anchor directions,
independent verification and an eight-second labelled temporal review.
**Not delivered:** real clean iris correspondence, reflection removal or scene
replacement. Original catchlights, crown sheen and lip gloss remain unresolved.
Photorealism is unmet.

## 1. Artist decisions and scope — preserve these

- Generic prepared footage remains the product. The reporter is bounded evidence.
- Reconstruct only from observations of the same iris where identity, material
  and visibility are reliable. Models supply coordinates, never performer RGB.
- Exact keyed-source fallback, including original catchlights, is accepted.
- Tracking plus scene-light replacement remains the full task. This round's
  synthetic coordinate result does not authorize either removal or replacement.
- No key retuning, generated performer texture, weaker gates or capture references.
- Full real reflection extent, pupil crossing, corneal pose, and camera/incident
  lid-lash-brow occlusion remain prerequisites to the reflection composite.

## 2. Frozen design and implementation

`docs/r18-evaluation-protocol.md` was written before new inference and scoring.
No model settings, points, motion or reflection strength were tuned afterward.

`scripts/reflection_motion.py` is a reusable array API for controlled additive
reflections over observed display RGB. `scripts/audit_r18_motion.py --shot` is
the reporter-specific experiment. `scripts/verify_r18.py --shot` independently
checks synthesis, coordinates, metrics, evidence and source fallback.

The observed texture is the two native 64x64 source-977 eye crops, with the
first and fourth iris anchors from the unchanged round-14 sparse labels:
left (23,37), (35,37), right (23,37), (37,36). Each point runs separately.
These labels are assistant annotations on seen footage, not independently
certified clean material. The base crop retains its original photographed
highlights; it is not reflection-free iris texture.

Each of four points has +x and +y sequences: **eight sequences of 24 frames**.
The entire observed crop translates by floor(t/4) native pixels with exact
integer sampling and reflected borders. All queried footprints remain interior.
An added neutral reflection moves the opposite way: centre
`q + (15-t)*axis`, overlapping material truth at t=12. It is a separable
raised-cosine patch of 3 px half-width/height, peak addition 0.6 in linear light.
Cached display sRGB is decoded before addition and re-encoded for the model.
Unaffected pixels remain bit-identical to the base control. There are **19
clipped channel values across the eight unique sequences** (0–7 per sequence),
reported in the cache and audit; this is not an exclusively unclipped test.

Both base-only and contaminated sequences run with anchors at t=0 (no added
reflection at the query) and t=12 (under the added reflection). This produces
**32 independent single-query cases**. CoTracker sees the full sequence; direct
RAFT pairs evaluate 0,4,8,12,16,20,23 excluding the anchor. CoTracker is scored
on the same six frames. There are **48 attempts per condition/anchor/model**,
384 total model observations. Query frames are never scored.

The complete 2x2 native anchor footprint is scrubbed before resizing, including
the reverse RAFT input. Surrounding reflection pixels remain visible by design.
Every case is fully recomputed with black and magenta substitutions under that
mask. Hidden-RGB invariance is about those four contributors, not hidden context
or the upstream fixed source crop. No photometric score or donor repair is run.

Released RAFT large C_T_SKHT_V2 and CoTracker3 scaled offline adapters, weights
and settings match round 17 exactly. No dependency or checkpoint changed.

## 3. Results — geometric evidence only

EPE is distance to the known translated material point, in native pixels.
Acceptance is unchanged: RAFT cycle <=0.5 px with complete native bounds;
CoTracker visibility with the same native bounds. An accepted error >1 px is
reported explicitly. Rows use identical attempted frames within each comparison.

| Input | Anchor | Model | Accepted / attempted | Median EPE, all attempts | p95 EPE | Accepted EPE >1 px |
|---|---:|---|---:|---:|---:|---:|
| Base only | 0 | RAFT | 48/48 | 0.0231 | 0.0684 | 0 |
| Added reflection | 0 | RAFT | 48/48 | 0.0254 | 0.0673 | 0 |
| Base only | 12 | RAFT | 48/48 | 0.0181 | 0.0401 | 0 |
| Added reflection | 12 | RAFT | 48/48 | 0.0211 | 0.0399 | 0 |
| Base only | 0 | CoTracker | 0/48 | 0.1870 | 2.3773 | 0 |
| Added reflection | 0 | CoTracker | 0/48 | 0.2252 | 2.1869 | 0 |
| Base only | 12 | CoTracker | 3/48 | 0.2967 | 1.1403 | 0 |
| Added reflection | 12 | CoTracker | 4/48 | 3.1517 | 11.3792 | 3 |

RAFT follows the known material path through this particular addition in both
anchor directions. The experiment does not show that an observed-iris anchor
is necessary for RAFT in this easy geometry: both directions work here.

CoTracker's low base-control acceptance makes this a poor setting for claiming
useful visibility or coverage. Its masked single-query control rejects 93/96
observations even without an added reflection. Do not interpret those rejections
as correct occlusion decisions; the translated base point is geometrically
visible. No unmasked ablation was run, so the cause of abstention is unproven.

The three accepted wrong CoTracker predictions all belong to **one sequence**,
left_0_y anchored at 12, at frames 0/4/8. Their material errors are
3.1278/3.1968/3.0987 px; reflection-centre errors are
17.8078/12.9052/7.8785 px. These are correlated observations, not three independent
failure events. The other accepted contaminated prediction is left_0_x at 23,
with EPE about 0.20 px. Accepted median contaminated anchor-12 EPE is 3.1133 px.

**Zero predictions meet the frozen accepted reflection-lock criterion:**
accepted, material EPE >1, reflection EPE <=1, centres separated by >=3 px.
There are 40 separation-eligible frames per anchor-0 row and 48 per anchor-12
row. The wrong CoTracker predictions cannot be described as proved reflection
lock-on merely because the added reflection changes the result.

### Limits that determine the next step

The entire eye crop, skin and original photographed highlights move together.
RAFT can use surrounding motion to get the right answer without establishing
real iris texture identity beneath a rotating cornea. There is one small added
reflection strength/shape, no independent lid motion, no real corneal rendering,
no unknown geometry, and no reconstructed RGB. The synthetic truth is translated
observed appearance, not real clean iris radiance. This is a positive controlled
case, not a reporter reflection fix or a model ranking on arbitrary footage.

This round deliberately completed the controlled identity test first. **A new
real context-control design was not preflighted.** Round 17's zero-of-42 result
under its fixed/+3px control remains the only real capacity finding. Do not run
another real photometric comparison until a frozen, discriminating design has
adequate common material and complete exclusions across every arm.

## 4. Verification

Audit, encode and final independent verifier exited 0. Verification records:

- Eight synthetic sequences checked against independent OpenCV integer warps,
  analytic reflection centres/support and scalar transfer/addition calculations.
  Four signed x/y analytic controls also pass; inference uses +x/+y only.
- All query footprints contain precisely the translated four source contributors;
  anchor 0 has zero added reflection there, anchor 12 has positive addition at
  all four contributors. This is known geometry, not inferred material masking.
- **384 model observations** independently checked, plus native bounds/visibility
  over all 768 saved CoTracker frame positions. Maximum independently resampled
  RAFT coordinate difference 0.00000190735 px; cycle difference 0.00000378543 px.
  All summary counts, all-attempt and accepted-error statistics match.
- **64 full inference substitution recomputations**, **1,152 exact array-hash
  comparisons**, covering both RAFT fields, all temporal outputs, exclusions and
  query inputs. Cached arrays match the recorded reference hashes.
- All **300 exact keyed-eye fallbacks** and **304 protected files** unchanged.
  Prior round-15/16/17 code hashes and model provenance match. Both input crop
  hashes and the unchanged sparse-label hash match.
- The round-14 search-isolation regression passes: 223 searched footprints,
  428 unchanged training pixels and selected angle; deliberately corrupted
  validation MAE 144.9259. This is a controlled regression only.
- All three new scripts compile. Audit/verifier without `--shot` list shots.
  `git diff --check` passes.

The first verifier invocation with a shot reached the existing fallback helper
and failed because the new verifier had not enabled OpenCV EXR support. Fixed
`OPENCV_IO_ENABLE_OPENEXR=1` before importing OpenCV; the full verifier then
passed. No inference data or criteria changed. The audit also prints the known
harmless git global-ignore permission warning, but exits 0.

## 5. Review and reproduction

Review: `music_videos/reporter_test/shots/reporter_a/final/r18_motion.mp4`.
**192 frames, 24 fps, 1280x790, CRF 12, eight seconds.** This is artificial motion
at a review rate, not the reporter's original temporal performance. Eight
one-second segments cover both points in each eye and both motion axes.

Each frame has base controls on the left and added-reflection cases on the
right, RAFT/CoTracker side by side; anchor 0 above anchor 12. Cyan crosses mark
known material, magenta crosses added-reflection centres, and green/red circles
model acceptance/rejection. Query frames are labelled unscored. RAFT is shown
only at evaluated frames, without interpolated tracks. CoTracker's other frames
are labelled outside the paired score. No green marker means clean iris.

All 192 decoded thumbnails were inspected in eight ordered contact sheets;
full decoded boards 12,23,24,132,191 were also inspected. The accepted wrong
left_0_y frame-0 prediction is visible above material truth; the late right_3_y
prediction drifts far from it and is rejected. `logs/r18_visual_review.json`
records these post-inference observations. This was frame/contact-sheet
inspection, not real-time playback or artist approval.

```powershell
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r18_motion.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r18_motion/review --fps 24 --out r18_motion.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r18.py --shot music_videos/reporter_test/shots/reporter_a
```

`--review-only` redraws the saved audit. Reports: `logs/r18_motion.json`,
`logs/r18_verification.json`, `logs/r18_visual_review.json`. Reproducible sequence
and model NPZs, review PNGs, decoded boards and sheets: `composite/r18_motion/`.
The existing source/model caches are required. Prior round-17 work was already
uncommitted at session start; this round did not edit its code or evidence.

## 6. Remaining work and starter prompt for round 19

The new positive fact is limited but useful: small synthetic reflection motion
does not necessarily prevent masked RAFT correspondence to observed material.
Do not rerun this same easy translation test as evidence of real iris identity.
Before further synthetic scoring, freeze a control separating local iris motion
from surrounding lid/skin motion, with observed RGB and explicit known geometry.
Keep the base-only control and report its failures. A future unmasked comparison
could diagnose CoTracker abstention, but cannot replace the hidden-RGB contract
of a repair experiment or be ranked as the same test without qualification.

Before new real RGB scores, preflight a control design with enough observable
material. Validate its ability to reject known wrong correspondences before
using real errors; don't choose controls by favourable real scores or expand
unknown labels after inference. Keep >=8 identical context pixels, >=3 spaced
donors, MAE <=3, wrong-map margin >=1 and MAD <=2. If this reporter cannot support
the design, another prepared clip remains valid. No capture reference is required.

> Read round18.md and the frozen protocols. The paired synthetic test uses
> observed eye RGB with whole-crop translation and an independently moving added
> reflection. RAFT accepts all 96 contaminated observations with median EPE
> 0.0254/0.0211 px for anchors 0/12; this easy geometry does not prove iris
> identity beneath real reflections. CoTracker rejects most base-control points;
> contaminated anchor-12 accepts four, three wrong in one sequence. No accepted
> prediction meets the frozen reflection-lock criterion. Move toward local
> material motion independent of surrounding skin/lids, and preflight an
> adequately powered real control design before RGB scoring. Preserve observed
> performer RGB, all-arm native exclusions, hidden-pixel invariance, unchanged
> gates, exact keyed fallback and coupled removal/replacement. Real clean
> coverage, pupil crossing, corneal pose and camera/incident occlusion still
> block scene reflections; hair/lip separation remains open. Produce a labelled
> temporal review and write round19.md without forcing a repair.
