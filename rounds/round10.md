# Round 10 handoff — full-source iris search still supplies no supported reflection repair

**Date:** 2026-09-08. Starts from round 9 and the round-4-equivalent production defaults.
**No visual fix for the original catchlights, crown sheen or lip gloss was delivered.**
Expanding the donor interval from 150 to all 1,200 source frames still gives zero confident
samples beneath the detected original reflections. The mask review exposes concrete failures
of treating a tracked aperture as iris material. No photorealism row becomes green.

## 1. Delivered and verified

Three isolated reporter diagnostics, each requiring `--shot`:

- `scripts/track_r10_source.py`: sequentially decodes all 1,200 original frames at
  60000/1001 fps and stores native eye crops plus a separate full-source landmark cache.
- `scripts/audit_r10_iris.py`: compares 150-frame versus 1,200-frame donor pools on the
  same target RGB, masks, geometry, four hidden discs and bounded affine recovery.
- `scripts/probe_r10_donors.py`: bypasses the 12-donor shortlist at three fixed target
  frames and tests all donors meeting the initial context-overlap requirement.

The full original is **20.02 seconds**, source frames 0–1199. Targets remain source
700–849, local 0–149. The target landmarks are the existing cached landmarks; only donors
outside that interval use fresh sequential VIDEO tracking. All 1,200 frames have a face
detection, but **this does not establish iris validity**. Fresh versus cached target iris
landmarks differ by median **0.133 px**, p95 **0.625 px**, averaged over ten iris landmarks
per frame. Retaining cached target geometry keeps this from confounding the two arms.

Both arms use 8-bit OpenCV-decoded original RGB, transformed with the existing keyer's sRGB
numerical inverse. This is an isolated correspondence experiment, **not a validated BT.709
scene-linear ingest implementation**. No colour/gain fit is added. Median per-frame raw/keyed
eye differences are **1.409 left / 1.717 right display levels**, p95 **1.677 / 1.855**. Thus
do not compare these scores with round 9 as though only the donor count changed there.
Within round 10 the RGB is identical between arms; the pool is the deliberate difference.

SHA-256 before/after matches for **304 protected files**: original plate, `shot.json`, cached
target landmarks, production compositor, 150 keyed foregrounds and 150 baseline PNGs.
No production keying, normals, albedo, lighting, geometry, rendering or compositor was changed.

## 2. Matched hidden-pixel controls: more support, slightly worse errors

Round 9's `recover` function is reused directly: the native four-contributor footprint guard,
scrubbed target RGB before statistics/selection/fit, conservative brightness exclusions,
bounded masked affine ECC, 12 shortlisted donors, best seven usable fits and at least three
donors per sample. Target +/-3 frames are excluded. Donors may remain adjacent to each other.
Both arms align; this is a donor-pool comparison, not fixed versus affine registration.

Eligibility yields **470 left / 316 right patches**, excluding original compact candidates
from four predetermined radius-three hidden discs. Compare the intersection of support,
with equal weight per supported patch; partial support is recorded in the JSON.

| Matched control | Left short | Left wide | Right short | Right wide |
|---|---:|---:|---:|---:|
| Matched supported patches | 368 | 368 | 220 | 220 |
| Matched grid samples | 9,649 | 9,649 | 6,176 | 6,176 |
| Median relative RGB absolute error | 7.00% | 7.11% | 6.11% | 6.49% |
| p95 relative RGB absolute error | 16.29% | 16.93% | 14.73% | 17.05% |
| Median display-channel MAE | 1.82 | 1.86 | 2.02 | 2.08 |
| p95 display-channel MAE | 3.16 | 3.32 | 4.00 | 4.59 |
| Patches above 5 display levels MAE | 2 | 4 | 5 | 8 |

Any-support coverage increases from **375 to 417** left patches and **231 to 268** right
patches. These larger sets are not the matched sets above. Across hidden tests, the wide arm
actually selects outside-interval donors in **1,511/2,762 left donor slots** and **748/1,730
right slots**; it is not merely reusing the original interval under a different name.

The unchanged confidence gate is median donor-context MAE <=2 display levels plus per-sample
donor MAD <=2. On hidden controls it accepts **1,125 left / 962 right wide-pool samples**,
versus **2,899 / 1,462** short-pool samples. The accepted wide patches have median display MAE
**1.56 / 1.67**, p95 **2.23 / 2.43**, with none above 5. Conditional agreement does not certify
clean iris texture or make the extra unconstrained support safe.

Adjacent changes in reconstruction residual on identical lower-hole pixels/pairs also fail
to improve: on **104 left / 87 right pairs**, median display residual-change MAE moves
**1.725 → 1.749 left**, **1.678 → 1.756 right**; p95 **2.826 → 3.142**, **3.052 → 3.365**.
Actual photographed change is subtracted first. This still includes grid-tracking error;
it is not a flicker result on repaired reflections.

## 3. Actual candidate locations: zero confident evidence, including a shortlist check

| Candidate evidence | Left short | Left wide | Right short | Right wide |
|---|---:|---:|---:|---:|
| Frames with compact candidates | 59 | 59 | 108 | 108 |
| Candidate grid samples, summed over targets | 2,630 | 2,630 | 4,568 | 4,568 |
| Samples with >=3 usable donors | 6 | 0 | 10 | 12 |
| Samples passing confidence | **0** | **0** | **0** | **0** |

The wide pool supplies at least three usable context fits in **54/59 left** and **100/108
right** candidate frames, so this is largely missing usable observations at the reflection
location, not universally failed registration. The 12 supported right samples occur at local
44, 46 and 71 and fail the context-error gate. No proposed reflection repair is written.

To probe the shortlist ceiling, the additional experiment fixes local frames **21, 79, 120**
before reading results and attempts every donor with sufficient initial overlap. It retains
the same affine bounds, footprint exclusions and donor masks, then requires **each donor's**
context MAE <=2. This is a stricter individual-context coverage probe, not the main predictor's
median-context confidence rule; it does not compute a recovery or donor-MAD confidence.

| Eye / local frame | Initial eligible donors | Usable context fits | Individual MAE <=2 | Candidate samples with even one usable donor |
|---|---:|---:|---:|---:|
| Left 21 | 995 | 47 | 0 | 0 |
| Left 120 | 836 | 197 | 26 | 0 |
| Right 21 | 907 | 22 | 0 | 0 |
| Right 79 | 890 | 160 | 23 | 0 |
| Right 120 | 822 | 148 | 11 | 0 |

Left local 79 has **no compact candidate**, despite a visible softbox reflection (see below),
so that probe abstains rather than inventing a mask. Displayed accepted donors for the other
targets still contain the softbox over the outlined target region. Bypassing shortlisting
does not expose clean samples at these tested targets under this gate. This is not proof
that every temporal, physical-eye or additive decomposition model is impossible.

## 4. Mask audit and complete review artefacts

Under `music_videos/reporter_test/shots/reporter_a/final/`:

- **`r10_full_source_masks.mp4`**: all 1,200 source frames; raw registered eyes on the left,
  overlays on the right; top row subject-left. Red = compact candidates, orange = conservative
  matching exclusion, dim = outside the estimated visible iris/aperture.
- **`r10_iris_withheld_comparison.mp4`**: all 150 target frames; truth, short pool, wide pool.
  Only the fixed known lower control disc is substituted. Purple means unavailable evidence.

Both are CRF 12 at the source **60000/1001 fps** and decoded completely: **1,350 frames**.
Every decoded frame was inspected in ordered sheets: twenty 60-frame mask sheets and five
30-frame comparison sheets, with each thumbnail resized to 246x180. Enlarged mask PNGs at
source **120, 200, 767, 779**, comparison PNGs at local **67, 120**, five donor panels, and
native source crops at **0, 300, 600, 721, 900, 1199** were also inspected.
**Ordered-frame inspection, not real-time playback or artist approval.**

Concrete failures against the images, not just measurements of the masks themselves:

- Source **120 and 200**: nominal iris/aperture regions contain extensive sclera and skin;
  brightness exclusion leaves some of it available. `found=True` is not material evidence.
- Source **767** (local 67): right compact candidates land along the lower sclera/rim while
  the upper softbox is visibly present. The estimated aperture clips the upper reflection.
- Source **779** (local 79): left compact candidates are empty despite the visible softbox;
  the broader orange exclusion does cover it. Zero candidates does not mean no reflection.

These are qualitative counterexamples, **not hand-labelled pixel truth or precision/recall**.
The full-clip review shows recurring misses and lower-rim detections. Do not relax the donor
exclusion to create coverage: the masks cannot yet distinguish iris, sclera, lid and reflection
reliably. The lower control can look plausible when supported, but there is no consistent
visual improvement from widening the pool and the original reflections remain throughout.

Evidence:

- `logs/r10_iris_audit.json`: matched controls, candidate failures, per-patch donor IDs,
  temporal residuals, full-source mask counts, code checks and decode records.
- `logs/r10_donor_probe.json`: exhaustive fixed-target probe counts and shown donor IDs.
- `logs/r10_visual_review.json`: actual inspection method and frame-specific mask findings.
- `composite/r10_iris/`: original eye crops, isolated landmarks, full affine/hash evidence,
  review PNGs, decoded contact sheets and donor panels; regenerable and gitignored.

Short-arm donor indices are local to source 700–849; wide-arm IDs are original source indices.
The shared `review_video.json` describes the new hidden-control video; each video retains its
own encode-provenance sidecar. Re-running diagnostics writes inspection as pending.

## 5. Checks, reproduction and settled limits

Passes: four known-direction registration checks (**0.0061–0.0258 grid px** centre error),
native-footprint independence, **ten real hidden-RGB perturbation checks** including supported
wide predictions, missing-evidence abstention, full decode and 304 protected-file hashes.
All three scripts compile. An initial EXR read failed because the environment flag was set
after importing OpenCV; fixed import order, then reran the complete audit successfully.
The implementation checks pass; the reflection-recovery experiment does not.

```powershell
& dependencies/facetrack/.venv/Scripts/python.exe scripts/track_r10_source.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/audit_r10_iris.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/probe_r10_donors.py --shot music_videos/reporter_test/shots/reporter_a
```

Keep round-4-equivalent defaults. Original reflection removal plus actual lid/brow/lash source
occlusion are jointly required for replacement. No new corneal render or lid-only compositing
patch would meet that prerequisite. Hair/lips have received no new treatment. No on-set
reference, generative performer, key-threshold change or persistent Cycles data.

## 6. Starter prompt for round 11

> Read round10.md and its mask-counterexample images before proposing another eye repair.
> The full 20-second original has now been searched: a 1,200-frame pool gives zero confident
> candidate recovery, and an exhaustive fixed-target probe does not rescue it. Do not repeat
> pool expansion, lower brightness exclusions or call darker donors clean texture. The current
> iris/aperture masks admit sclera/skin and miss real glints; any further eye work must first
> validate material visibility and correspondence on native images, preserve hidden-footprint
> controls and prove evidence at the actual source reflections. A physical eye/reflection
> model is a new experiment, not an established solution. Consider moving to the still-open
> material-aware temporal hair/lip problem rather than another generic iris-median variant.
> Keep production unchanged until integrity and temporal evidence pass. Produce and inspect
> complete comparison video, state all unresolved complaints plainly, and write round11.md
> with a starter prompt for round 12.
