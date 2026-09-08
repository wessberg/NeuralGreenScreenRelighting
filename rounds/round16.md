# Round 16 handoff — temporal visibility works on closure stress, clean repair unproven

**Date:** 2026-09-08. Starts from round 15; production defaults remain unchanged.
**Delivered:** reusable released CoTracker3 adapter, frozen source-only material
regions, comparison with the retained RAFT baseline, controlled occlusion tests,
native sampling and hidden-RGB verification, temporal review video.
**Not delivered:** certified clean donor radiance, complete reflection removal,
real corneal/occluder validation or scene-light reflection replacement. Original
catchlights, crown sheen and lip gloss remain unresolved. Photorealism is unmet.

## 1. Artist decisions and scope — preserve these

- Generic prepared footage is the product; reporter audits are bounded evidence.
- Models supply coordinates/visibility, never generated performer appearance.
- Use same-eye observed RGB only where correspondence and material are reliable.
  Keep exact keyed source, including its original reflection, otherwise.
- Removal and scene replacement remain coupled. Visibility predictions alone
  cannot authorize either. No relaxed gates, key retuning or capture references.
- Corneal pose, pupil crossing, full reflection extent, and camera/incident-light
  lid/lash/brow occlusion still precede enabling the Cycles reflection composite.

## 2. Delivered implementation and frozen evaluation

`scripts/temporal_eye_tracker.py` uses the authors' **CoTracker3 scaled offline**,
checkout `dependencies/co-tracker` at commit
`82e02e8029753ad4ef13cf06be7f4fc5facdda4d`. The adapter verifies that revision and
rejects tracked local changes. Checkpoint:
`dependencies/cotracker-weights/scaled_offline.pth`, 97.2 MiB, SHA256
`2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`.
The hash pins the downloaded evaluation artifact; it is not a separately
authenticated publisher signature. The download came from the URL in the
[authors' hub configuration](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/hubconf.py).
The existing EllSeg runtime needed **no package installs or upgrades**.

The unchanged predictor runs six float32 iterations on CUDA with its 6x6 support
grid, backward tracking, and visibility >0.9. Its native model canvas is 384x512.
The adapter letterboxes to preserve physical pixel aspect, scrubs explicit native
exclusions before resizing, and returns native coordinates plus full-footprint
validity separately from the model's visibility. No reporter coordinates live in
the adapter. It accepts arbitrary native video arrays and t/x/y query points.
Physical-aspect letterboxing is our preprocessing choice, not a benchmark result
claimed for the authors' original demo preprocessing.

**The predictor overwrites query-frame coordinates and visibility.** Source 977
is therefore excluded from scoring, and its review panel says “forced visible”.
Do not use an exact query-frame track as an accuracy or visibility measurement.

`docs/r16-evaluation-protocol.md` and `scripts/data/r16_material_labels.json` were
written before CoTracker inference. Labels cover source 974, 978, 981, 982, 984,
985, 990, 994, both eyes. Their source-only nearest-neighbour grids were inspected
before new predictions. These are **new assistant region annotations on reused
footage**, independent of predictions but not an independent annotator. They do
not certify absence of weak reflection. CoTracker and RAFT use the same ten
reused round 14 iris queries anchored at 977 and round 15's 950–999 tracked crops.
The fixed upstream tracker/crop seeds remain outside hidden-RGB guarantees.

`scripts/material_footprints.py` rasterizes explicit iris, pupil, sclera,
skin/lash and reflection polygons, excludes a 1.5-native-pixel boundary band,
and requires all four bilinear contributors to lie in a labelled interior.
Unlabelled or boundary footprints are **unknown**, never clean or negative by
default. Iris compatibility is neither same-point identity nor clean radiance.
All ambiguity is retained in denominators.

The reporter audit compares saved unmasked RAFT direct maps with unmasked
CoTracker offline tracks. The same saved neural material proposal is applied to
both, and reproduces every relevant round 15 candidate-footprint result exactly.
RAFT cycle+proposal here means those two point tests, not the complete round 15
context/donor/eye gate. No photometric gate was retested or weakened.

## 3. Real results — visibility is useful, material support remains limited

All rows contain **80 attempts**: eight labelled frames, two eyes, five queries.
The material columns describe accepted predictions only.

| Arm | Accepted | Iris-compatible | Explicit incompatible | Unknown | Closure false-visible / 20 |
|---|---:|---:|---:|---:|---:|
| RAFT cycle <=0.5 px | 27 | 3 | 5 skin/lash | 19 | 5 |
| RAFT cycle + proposal | 14 | 2 | 0 | 12 | 0 |
| CoTracker visibility + in-bounds footprint | 36 | 7 | 0 | 29 | 0 |
| CoTracker visibility + same proposal | 20 | 5 | 0 | 15 | 0 |

Source-only inspection labelled **981/982 both eyes** as no resolvable iris.
CoTracker rejects all 20 queries there; cycle alone admits five. The combined
RAFT/proposal baseline also rejects all 20, so this is not a demonstrated
improvement over that combined baseline's closure rejection. The five cycle
acceptances also independently land inside labelled skin/lash interiors.

These are correlated points from two adjacent frames of one blink, not 20
independent occlusion events. Source blur limits the closure annotation; no
dense true material trajectories or visible-point recall labels exist here.
**No real tracking endpoint accuracy or clean-radiance accuracy is established.**

At narrowed source **985 left, CoTracker marks 2/5 visible**, while the two
accepted footprints remain unknown under frozen material labels. Full-board
review confirms the early visibility return during reopening. This is a
post-inference visual concern, not a newly relabelled false-positive statistic.
At 984 both eyes and 985 right it keeps all five queries rejected. Do not
reinterpret its 0/20 closure statistic as reliable material visibility everywhere.

Conservative iris regions admit only **8, 10, 17, 19, 26, 23** complete native
footprints on the six annotated open eyes (974, 990, 994). Most of the crop is
skin, unlabelled, or boundary. Region counts are before query/control exclusions.
This round did not reuse the underpowered eight-context-pixel test as progress.
The new labels are sufficient for a bounded closure/material diagnostic, **not
for a certified, adequately powered clean-donor photometric benchmark**.

All 12 original reflection probes still have round 15's zero candidate support;
this round did **not** rerun them with CoTracker. Tracking lower-iris queries
through a blink does not establish observations beneath those reflections.

## 4. Controlled tests and verification

Four controlled 24-frame videos use observed eye RGB, known horizontal or
vertical translations spanning -3..+3 native pixels, square/anamorphic aspects,
and a fully opaque rectangle during frames 8–15. Four interior query points
have exact known geometry and occlusion. The query frame is excluded.

- **240/240 visible point-frames detected**, **0/128 occluded false-visible**,
  and **128/128 post-occlusion recovery point-frames detected** across four tests.
- Median visible native endpoint errors: **0.03411, 0.03306, 0.01915, 0.02961 px**.
  Respective p95: **0.06696, 0.05690, 0.03197, 0.05706 px**.
- These are controlled geometry/visibility checks on an observed image with an
  artificial occluder, not real iris deformation or real clean-material proof.

`scripts/verify_r16.py --shot` passes:

- Eight analytic +/-x, +/-y transforms at both pixel aspects: maximum native
  error **0.000002394 px**; exclusions scrubbed before resizing.
- Independent double-precision bilinear resampling of **500 saved real native
  coordinates**: maximum difference **0.0**. Complete footprint masks and every
  scored material class independently recomputed contributor by contributor.
- Both real masked sequences recomputed with black and magenta replacements at
  all five anchor query pixels: **four exact substitution recomputations** of
  coordinates, visibility and native validity. Masked runs are separate from
  unmasked visibility results. They assert excluded-RGB invariance only; no
  withheld radiance/context claim is made for surrounding model input.
- All **100 source crop hashes**, **300 exact keyed-eye fallbacks**, and **304
  protected hashes** unchanged. The three round 15 recorded code hashes match.
- Round 14 search-isolation regression still passes: 223 searched footprints,
  428 unchanged training pixels, unchanged selected angle, changed validation
  MAE 144.9259. No earlier search/sampling path was modified.
- Report metrics, protocol/annotation/checkpoint hashes, synthetic occlusion
  extent and truth, query-frame exclusion, script compilation verified.
  Audit/verifier without `--shot` list available shots. `git diff --check` passes.

The first audit shell reported exit 1 because PowerShell treated git's harmless
user-ignore-file permission warning on stderr as an error record. Its log has
complete inference/results and no Python traceback; the independent verifier
completed with exit 0. Do not rely on that initial shell status as run evidence.

## 5. Review and reproduction

Review: `music_videos/reporter_test/shots/reporter_a/final/r16_visibility.mp4`.
**50 frames, 60000/1001 fps, 1280x736, CRF 12**, about 0.834 seconds. Panels:
source / RAFT cycle+proposal / CoTracker visibility / exact source fallback.
The video is labelled **NO REMOVAL / REPLACEMENT**. CoTracker green marks only
predicted visibility; source and fallback retain the original softboxes.

All 50 ordered decoded thumbnails were inspected, plus full boards at 977, 981,
982, 984, 985, 990, 994. This was contact-sheet/full-frame review, not real-time
playback or artist approval. Decoded video hash and visual review are recorded
separately in `logs/r16_verification.json` and `logs/r16_visual_review.json`.
Scores are in `logs/r16_visibility.json`. Reproducible NPZ/PNG caches live under
`composite/r16_visibility/`. `logs/review_video.json` now describes this review.

```powershell
git clone https://github.com/facebookresearch/co-tracker.git dependencies/co-tracker
git -C dependencies/co-tracker checkout 82e02e8029753ad4ef13cf06be7f4fc5facdda4d
# Fetch scaled_offline.pth from the URL in temporal_eye_tracker.py; hash is pinned.
& dependencies/ellseg-runtime/.venv/Scripts/python.exe scripts/audit_r16_visibility.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r16_visibility/review --fps shot --out r16_visibility.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe scripts/verify_r16.py --shot music_videos/reporter_test/shots/reporter_a
```

Use `--review-only` on the audit to redraw cached tracks. Existing round 10 source
crops and round 14/15 evidence are required. No production renderer/compositor,
scene, original footage, prior neural checkpoint or package changed.

## 6. Remaining work and starter prompt for round 17

Keep both pretrained baselines. Visibility has now been exercised, but material
identity, weak reflection contamination and complete source-removal coverage
remain the bottlenecks. Before another real photometric comparison, freeze a
single-query-at-a-time design and preflight label/context footprint capacity
without consulting errors; report insufficient cases explicitly. Do not silently
relabel this round's unknowns, lower the eight-pixel gate, or certify donor RGB
from visibility. A different prepared clip with resolvable iris detail remains
valid future evaluation; the reporter failure is not universal impossibility.

> Read round16.md and the frozen protocols. CoTracker3 now runs and rejects all
> 20 source-labelled closure queries, but RAFT+proposal does too. Its 36 accepted
> labelled-frame predictions have only seven conservative iris-compatible
> footprints and 29 unknowns. Source 985 left accepts two unresolved points.
> Controlled occlusion success is synthetic evidence only. Move toward testing
> actual reflection-region donor material and same-point identity; do not spend
> another round only counting lower-iris tracks. Design and preflight adequately
> observable context with frozen source-only labels and one hidden query at a
> time before scoring. Preserve all-arm native footprints, hidden RGB, unchanged
> gates and exact keyed fallback. Reflection clean truth, full removal coverage,
> pupil crossing, real corneal pose and camera/incident lid-lash-brow occlusion
> still block coupled scene-light replacement. Hair/lip separation remains open.
> Produce a labelled temporal review, distinguish new/reused/post-hoc evidence,
> and write round17.md. No generated performer texture or forced repair.
