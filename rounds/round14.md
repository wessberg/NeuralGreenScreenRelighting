# Round 14 handoff — neural correspondence measured, no reflection repairs earned

**Date:** 2026-09-08. Starts from round 13; production defaults remain unchanged.
**Delivered:** reusable experimental neural-proposal correspondence, frozen primary
evaluation, fresh source-only sparse labels, search-wide interpolation exclusions,
verified source fallback, and a diagnostic review video.
**Not delivered:** validated clean donors, reliable blink/visibility classification,
complete source-reflection removal, or scene-light reflection replacement. Original
catchlights, crown sheen and lip gloss remain unresolved.

## 1. Artist decisions and scope — preserve these

- Generic prepared footage remains the target; this reporter experiment does not
  establish generalization. Preserve each shot's own camera/color metadata.
- Use evaluated pretrained components before inventing another contrast detector.
  EllSeg supplies complete-ellipse geometry; RITnet supplies visible-part proposals.
  Neither certifies clean radiance or the material beneath a reflection.
- Reconstruct the same eye from observed performer pixels only. No generated
  performer texture, new on-set reference requirement, or reporter key retuning.
- Tracking, source removal **and scene-light replacement** remain the full task.
  Unreliable eyes keep exact keyed source, including the original catchlight.
  Only permit replacement when complete source removal is independently validated.

## 2. Delivered implementation and evaluation separation

`scripts/neural_iris_correspondence.py` is independent of reporter coordinates and
annotations. It consumes native display RGB, same-eye donor rows, pixel aspect,
exclusions and fixed crop tracking. EllSeg native curves become a concentric
pupil/limbus material sampler; pupil-center displacement, candidate area and visible
pupil area provide **experimental, uncalibrated** abstention rules. The combination
of eroded neural iris proposals is still not a validated material mask.

Target exclusions are scrubbed before both neural models, and the complete target
proposal, shortlist, fit, validation and reconstruction are recomputed. Fitting
uses one angular sector, validation another. Both fixed and fitted-torsion arms
start from the same fixed-map shortlist, ranked by training error alone. Donor
query/validation interpolation footprints are excluded **across the entire torsion
search**, not merely for the winning candidate. Returned RGB is sampled original
donor RGB. No renderer or production compositor was changed.

The primary rules were written in `docs/r14-evaluation-protocol.md` before viewing
new boards or model results. Fresh assistant labels in
`scripts/data/r14_native_labels.json` were made from source-only native grids for
959, 977 and 995, both eyes, before neural overlays. These are not independent human
ground truth. Earlier rounds already processed this same clip; this is new
annotation and withheld-pixel evaluation, not unseen footage.

`scripts/audit_r14_neural.py --shot` is explicitly reporter-only. The reusable
correspondence API has no reporter coordinates. The audit uses:

- **163 donors per eye:** source 0–699, stride 4, excluding ±8 around old annotation
  frames 79, 217 and 521. No target is a donor.
- **50 target frames:** 950–999. Two fixed five-pixel holes per eye on every other
  frame: **100 attempted holes / 500 native pixels**, including abstentions.
- Inherited fixed context gates: training/validation MAE ≤3 display code values,
  wrong-rotation control worse by ≥1 on identical valid pixels, at least eight
  control pixels; reconstruction needs three spaced donors and MAD ≤2.
- Six separate sparse real-reflection probes. Their hidden clean truth is unknown;
  their visible RGB is never treated as clean reconstruction truth.

No thresholds were tuned on the new samples. Training here means local donor
correspondence fitting; neural weights were not trained or fine-tuned.

## 3. Real findings — support is not reliable correspondence

| Primary candidate-region metric | Fixed coordinates | Fitted torsion |
|---|---:|---:|
| Supported holes / 100 | 19 | 19 |
| Supported pixels / 500 | 59 | 61 |
| Paired supported pixels | 57 | 57 |
| Median per-hole MAE on paired pixels, display levels | 4.2344 | 4.3501 |
| p95 per-hole MAE on paired pixels | 12.0550 | 14.5384 |
| Median per-hole relative RGB error, paired | 5.4289% | 6.5579% |
| Pixels passing context + agreement | **0** | **0** |

The 496 retained fits across both arms include **zero passing the full individual
context gate**. Torsion does not improve the paired result. These scores cannot be
compared numerically with round 12's results because targets and donor pools differ.

**The primary score is not certified clean-iris accuracy.** Only 14/100 fixed
holes lie wholly inside the original combined candidate mask; among the 19 holes
with support, only three do. Hole locations were deliberately fixed independently
of neural acceptance, and all attempts remain in the denominator.

That finding motivated an explicitly **post-hoc supplementary analysis** of all
30 previously frozen manual iris points. All five points in each eye image were
withheld together; donor pool and thresholds stayed unchanged. Five of six eye
images abstained after masked inference. Only source 995 right had support: four
pixels for fixed coordinates (MAE 6.8613) and three for torsion (MAE 9.7155).
Again **zero pixels passed the combined gates**. These arms have different support;
their unpaired errors are not an improvement ranking. The report also retains
paired scores. Annotation uncertainty remains 1.5 native pixels.

Fresh sparse semantics: RITnet admits **19/30 iris points**, the combined eroded
mask **14/30**; both admit **0/30 noniris** and **0/12 marked reflection points**.
This is sparse evidence only, not proof that all donors exclude reflections.

Unmasked candidate gates accept 42/50 left and 40/50 right target eyes. Contact-sheet
inspection exposed an important counterexample: **source 984 left is nearly closed
but passes** with 51 candidate iris pixels, 27 pupil pixels, center offset 0.5696 px
and pupil ratio 0.2019. Its inferred iris cannot be certified from that view. This
is post-hoc visual evidence, not a new label used to retune the gate. Source 985
left/right correctly reject under the area rule, despite residual mask proposals.

All **six real-reflection probes / 12 marked points have zero donor support** in
both arms. Each highlight approaches or overlaps an obscured pupil boundary:
pupil crossing cannot be excluded. Sparse bright points are also not complete
reflection masks. Iris-only replacement must abstain even if a subset later gains
support; the unknown pupil and full reflection extent still need evidence.

## 4. Verification and review artifacts

`scripts/verify_r14.py --shot` passes:

- Native ellipse mapping tested at four rotations and square/anamorphic pixel
  aspects: maximum limbus radial residual **1.3323e-15**.
- Synthetic known-geometry search-isolation regression poisons all 223 searched
  query/validation footprints. The selected angle and 428-pixel training score
  remain exact while validation MAE changes to 144.9259. This verifies exclusion
  mechanics, not real detection or repair quality.
- All **100 primary holes**, including 19 with positive support, retain exact
  predictions, donor selection, gates and metadata under black and magenta hidden
  RGB substitutions. The six supplementary labelled-eye tests also pass both
  substitutions. This excludes the upstream fixed face tracker and crop seed.
- Search-wide fitting/validation/query native footprint assertions execute on all
  retained real fits and reflection probes.
- All **300 saved keyed-eye crops** still equal their original keyed frame regions;
  fallback through `apply_eye_repair` changes **zero channels**.
- All **304 protected hashes** from round 10 match. New scripts compile; both
  command-line stages list available shots when `--shot` is omitted.
- Diagnostic video fully decodes: **50 frames, 60000/1001 fps, CRF 12**.

Review: `music_videos/reporter_test/shots/reporter_a/final/r14_neural.mp4`.
Left: native source. Middle: candidate-mask/gate diagnostic. Right: exact source
fallback. It is 0.834 seconds of source motion, **not a reflection-fix video**.
The video uses source-space crops; exact keyed-space fallback is checked separately
on the existing 300 keyed crops. Nothing was mapped into a new keyed repair path.

Visual inspection covered all three source annotation boards, all three fresh
overlay boards, all 50 ordered video thumbnails, and full boards for 982/984/985.
The middle mask changes abruptly during the blink; the source reflection remains
on both source/fallback panels. This was image/contact-sheet review, not real-time
playback or artist approval. See `logs/r14_visual_review.json`.

An initial local implementation excluded footprints only per hypothesis; it was
strengthened to the full-search union before the final reported run. The synthetic
regression prevents that selection-leakage bug from returning. The frozen thresholds
and sample selection were not changed. Earlier preliminary console numbers are
superseded by `logs/r14_neural.json`.

```powershell
& dependencies/ellseg-runtime/.venv/Scripts/python.exe scripts/audit_r14_neural.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r14_neural/review --fps shot --out r14_neural.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe scripts/verify_r14.py --shot music_videos/reporter_test/shots/reporter_a
```

Use `--review-only` on the audit to redraw its cached diagnostic without rerunning
inference. The verifier leaves visual approval pending; a rerun cannot imply artist
approval. The pretrained runtime and checkpoints are unchanged from round 13.

## 5. Remaining work and next decision

This evaluated combination of framewise neural proposals, concentric ellipse
coordinates and fitted torsion has **not demonstrated reliable real correspondence**.
Do not relax gates to turn donor agreement into success, or treat failure here as
proof that temporal reconstruction cannot work on other footage.

Needed evidence is now specific: stable visible-material tracking through blinks,
independently clean donor support, and correspondence that survives masking on
known iris pixels. Use pretrained temporal/dense correspondence components for a
bounded comparison before adding more custom ellipse fitting. Any future evaluation
must declare which samples are new, reused or post-hoc; this round's labels/results
are now seen data.

The Cycles renderer remains the scene-light reflection source. Before enabling it,
its real-shot corneal pose, lid/lash/brow visibility and incident-light occlusion
must be solved and checked. The existing smooth proxy occluder is not an eye socket,
and the rejected flat compositing cap still has no validated replacement. These
were reviewed in code only; no new render/occlusion result is claimed this round.
Hair sheen and lip gloss received no treatment.

## 6. Starter prompt for round 15

> Read round14.md and the frozen protocol. EllSeg/RITnet proposals plus concentric
> material coordinates and torsion earn zero reliable real repairs; the candidate
> gate accepts a nearly closed eye. Primary fixed-hole errors are not certified
> clean-iris accuracy; the supplementary frozen-label test also fails the gates.
> Keep source-keyed fallback and coupled removal/replacement. Evaluate a pretrained
> temporal/dense correspondence or visibility component against known observed iris
> pixels and blink failures before more custom fitting. Separate new evaluation
> from reused/post-hoc samples; preserve neural hidden-RGB exclusion and search-wide
> native interpolation controls. Do not feed generated texture into performer
> output or force pupil-crossing repairs. Tracking, complete source removal and
> actual scene-light replacement remain the task. Retain corneal pose and
> lid/lash/brow/incident-source occlusion in the work plan. Produce an honestly
> labelled review video and write round15.md with the actual remaining blockers.
