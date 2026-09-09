# Round 33 handoff — one sequence-wide eye estimate; measurably better, not yet accepted

**Date:** 2026-09-09. Starts from round 32. Production defaults unchanged.
**Delivered:** the measurement that makes an observed-pixel sequence repair possible at
all, a registered sequence-wide appearance atlas per eye, a repair driven by it over all
150 frames, contact sheets covering every frame, a withheld-pixel evaluation across a
temporal split, a flicker measurement that separates motion from jitter, and
verification. **Not delivered:** artist acceptance, a clean pupil, scene-light
replacement, or any production change. Photorealism is not met.

## 1. Artist steering

He rejected round 32 in these terms, and added mid-round:

> yeah, it wasnt tempirally stable in terms of neither shape nor color, but also the
> pupil was sometmes brown

Three distinct complaints — **colour** instability, **shape** instability, and a **brown
pupil** — and this round is organized around them. The round-32 requirement to put
frames in front of him constantly and to inspect every frame on contact sheets was
followed; two of this round's four significant bugs were caught only by looking at
frames, and one of them was invisible in the logs.

## 2. The measurement that had to come first

Round 32's transferable finding was that a next attempt needs the eye estimated **once
over the sequence in a stabilized frame**. That is only possible if the corrupted
footprint MOVES in the eye's own coordinates, so every cell it covers is observed clean
in other frames. `eye_sequence_occupancy.py` measures exactly that and nothing else
(`logs/r33_occupancy.json`), reusing round 32's polar frame and detector unchanged:

| | left | right |
|---|---:|---:|
| footprint radius rho, range over the shot | 0.383 → 0.749 | 0.284 → 0.741 |
| arc travel in eye coordinates | 50.6 px | 22.0 px |
| grid cells ever corrupt | 507 | 621 |
| **corrupt cells with ZERO clean observation** | **1** | 71 |
| clean observations per corrupt cell, median | 24 | 7 |

**The highlight sweeps.** 506 of 507 ever-corrupt cells left, and 550 of 621 right, are
observed clean elsewhere in the shot. There is real observed material behind the
highlight. This is not what the round 8–17 donor searches asked and does not contradict
them: they looked for correspondence between individual pixels, and this pools in the
eye's own coordinates, where no per-pixel correspondence is required.

## 3. What was built

`eye_sequence_atlas.py` estimates one appearance atlas per eye for the whole shot, and
`eye_atlas_repair.py` uses it. Per eye:

1. Sample every frame's eye onto one 97×97 grid in normalized eye coordinates, with the
   detected footprint and a two-pixel collar excluded.
2. **Register** each frame to the atlas by translation and isotropic scale, bounded
   absolutely at about one native pixel and re-zeroed after every rebuild.
3. Fit a **per-frame illumination field** — a constant plus a linear ramp in eye
   coordinates, per channel, by IRLS in the log domain — then smooth its coefficients
   over frame index.
4. Rebuild the atlas as a per-cell median of gain-normalized radiance. Iterate 5 times.
5. Interpolate the atlas across cells the sweep never uncovered, **once**.
6. Detect the footprint in each frame against that fixed prediction, vote the mask over
   five frames in eye coordinates, fill with `atlas × gain`, iterate the fill until no
   residue is left, and feather the edge.

The dark undershoot beside the highlight is still **not** repaired. It is coding, not
material, but the artist reads it as the pupil.

## 4. Measured findings

### 4.1 Withheld pixels, across a temporal split

Round 32's frozen rotation design, unchanged, so the numbers are comparable on identical
pixels. **The atlas that predicts a site in the second half of the shot is built only
from the first half, and vice versa** — scoring an atlas on frames it pooled would be
close to self-prediction, and the assumption under test is that ONE estimate serves the
whole shot. 162 sites scored, 5,739 / 1,599 pixels.

Median absolute display RGB error, 0–255 levels, identical pixels in every row:

| prediction | left | right |
|---|---:|---:|
| plate's own texture floor | 4.6646 | 5.2760 |
| **round 33 atlas (held half)** | **6.3662** | **6.5438** |
| round 32 repair | 8.9720 | 7.9791 |
| radial profile alone | 9.0444 | 7.8197 |
| annulus median | 13.3251 | 12.9904 |
| flat ring gain (round 31's shape) | 12.6648 | 13.9915 |

**29.0 % better than round 32 left, 18.0 % better right**, on pixels the atlas never
saw, and at **1.36× / 1.24× the plate's own texture floor** against round 32's 1.92× /
1.59×. Round 32's own p95 of 28.65 / 26.35 falls to 21.96 / 21.90.

### 4.2 Temporal stability

Round 32's adjacent-frame step is reported for comparability, but it is a poor measure
and the round-32 notes say why: it conflates real eye motion with flicker, which is how
round 32 could report a median below the source's own control while plainly flickering.
**Jitter** is the same fill colour measured against a five-frame rolling median of
itself, so a smooth drift as the eye turns costs nothing and a one-frame excursion costs
everything. The source's own pixels are the paired control in both.

| left / right | median | p95 | max |
|---|---:|---:|---:|
| repair jitter | 0.68 / 0.41 | 4.74 / 6.01 | 26.44 / 57.47 |
| source jitter (control) | 1.67 / 1.45 | 8.72 / 12.11 | 38.57 / 44.93 |
| repair step | 3.53 / 2.40 | 36.23 / 19.92 | 83.16 / 84.08 |
| source step (control) | 4.00 / 2.53 | 37.63 / 34.46 | 78.72 / 68.59 |

**The repaired region now jitters less than the source's own pixels do** at median and
p95 on both eyes, and at max on the left. Right max 57.47 exceeds its control: there are
still individual frames that jump.

The illumination field's own adjacent-frame step, measured over the iris where it is
applied, drops from raw p95 0.444 / 0.566 and max 2.408 / 2.792 to **smoothed p95 0.024
/ 0.026 and max 0.030 / 0.030** — 18× and 22× at p95, 81× and 92× at max. That is the
colour-flicker mechanism removed by construction rather than by tuning.

Fill display colour across the shot: left R 50.1–115.3, G 32.8–90.4, B 24.1–82.0;
round 32's was R 36.2–131.9, G 12.0–87.3, B 2.2–65.5. In ratio the blue channel goes
from a 30× traverse to 3.4×.

### 4.3 Shape stability is improved but NOT solved

Mask IoU between adjacent frames, in eye coordinates: **median 0.780 / 0.777, p5 0.185 /
0.228, minimum 0.0**. Footprint size changes by a median of 5 / 4 pixels between frames,
p95 23 / 21, max 43 / 56. So the mask is stable most of the time and still jumps
outright on a minority of frames. **This is the most likely remaining source of visible
instability and it is the first thing to attack next.**

### 4.4 Registration earns its place, and the amount matters

The limbus landmark's own jitter was measured before the search was sized: **0.353 px
rms / 0.578 p95 left, 0.256 / 0.521 right**, against a local quadratic trajectory fit.
So the correction wanted is well under one native pixel.

Paired, same frames: pooled-sharpness (mean gradient magnitude of atlas log-luma) rises
from **2.785 → 3.423 left and 2.185 → 2.761 right** with registration on, and the seen
frame clean-pixel error falls from 3.30 → 2.51 left and 4.09 → 3.13 right.

**An unbounded search made all of this worse and looked like it was working.** Mean NCC
FELL from 0.921 to 0.617 across iterations while total shift grew to 1.9 px rms and 3.0
px max — atlas drift, because the reference is rebuilt from the frames it just moved.
Two guards fix it: an absolute bound on shift from the landmark, and re-zeroing the mean
shift after every rebuild. Final shift is 0.91 / 0.95 px rms, mean NCC 0.922 / 0.918.

### 4.5 Coverage and abstention

141 of 150 frames repaired; 274 eye instances repaired, 26 abstain — 17 with no usable
eye frame (the two blinks), 8 for registration below threshold, 1 because the prediction
was not darker than the source. **16,232 pixels changed across the whole sequence.**
Peak display inside the footprint falls 0.8313 → 0.4157 left and 0.8510 → 0.3931 right.

## 5. Four bugs, each of which produced a plausible wrong answer

Recorded because none announced itself, and two were visible only in frames.

1. **Detection was gated on the atlas's support mask.** The highlight dwells in roughly
   one place for many consecutive frames, so its own cells are exactly the ones with
   fewest clean observations — only **7 of 85** footprint pixels survived the gate. The
   detector could not see the thing it exists to find, and the masks landed on the lid
   and limbus instead. Support is a constraint on what may be FILLED, never on what may
   be found.
2. **The repair left a light speck at the footprint edge** — round 31's rejected
   artefact, returning by a different route. Two causes, both found by looking: a 3-of-5
   temporal vote ERODES a mask, so the vote may only add to this frame's own detection,
   never subtract; and the atlas stopped at rho 0.88, so the outer edge of the highlight
   had no prediction. That limit was inherited from round 32's radial-profile model,
   which needs radial symmetry — **an atlas is a full 2D map and does not**, so it can
   hold the limbal ring honestly. Extending it to rho 0.98 took eyes with unsupported
   footprint pixels from 81 → 9 left and 35 → 0 right, and leftover residue p95 from
   20 → 8 and 10 → 6.
3. **At the blinks the fill made the eye BRIGHTER.** With the iris a sliver, the
   registration collapses (NCC 0.0–0.08) and the atlas lands in the wrong place. Caught
   by a verification assertion, not by eye. Two gates now abstain: NCC below 0.5, and a
   prediction that is not darker than the source over the footprint.
4. **The log reported removal as a no-op.** `patch` was a view into the output array, so
   the "source" peak was read back after the blend and both peaks printed the same
   number while the repair was working correctly in the pixels.

Also corrected mid-round: an early claim here that the pooled eye is "brown with no
pupil". That was an artefact of averaging on the jittering landmark and of reading a
radial average; single frames show a 12-px region at 35–70 display inside an iris at
90–130. The registered atlas is sharper than the unregistered pool by the measure in
§4.4, but **no sharp pupil disc localizes in it either**, consistent with round 32 §3.6.

## 6. What is still wrong

* **A light residue survives on 112 of 140 left eyes** (median 3 px, p95 8, max 27) and
  34 of 134 right. It concentrates on partly-lidded frames. Visible at 840–849.
* **The mask still jumps on a minority of frames** (§4.3), which is shape flicker.
* The repaired pupil reads as a dark smudge, not a clean pupil, because the atlas has no
  resolvable pupil to offer.
* 1,096 of 1,405 withheld sites are still rejected for self-overlap — round 32's design
  weakness, inherited unchanged so the numbers stay comparable.
* No scene-light replacement. Rows 1–5 of the constraints table are untouched.

## 7. Verification

`logs/r33_verification.json`: 150 frames checked, **311,023,768 pixels bit-exact source
outside the reported mask**, every written source frame identical to the keyed plate,
`changed_pixels` matching the mask, and **all 274 repaired eyes darker than their
source**. The standing **300 exact keyed-eye fallbacks and 304 protected hashes pass**
(`verify_r15.fallback`, reused unchanged). No production module was modified;
`production_enabled` and `artist_accepted` are false.

Reviews, display-sRGB CRF 12, not masters and not acceptance:

- `final/r33_repair_eyes_slow.mp4` — 150 frames at 12 fps, source left / repair right,
  eye band `848 204 265 178` at 4× nearest neighbour. Same crop and rate as the file the
  artist rejected in round 32.
- `final/r33_repair_eyes.mp4` — the same at the shot rate.
- `composite/r33_repair/sheets/contact_{000,025,050,075,100,125}.jpg` — every frame,
  both eyes, source above repair with the mask outlined.

```powershell
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_sequence_occupancy.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_sequence_atlas.py --shot music_videos/reporter_test/shots/reporter_a --count 150
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_atlas_repair.py --shot music_videos/reporter_test/shots/reporter_a --count 150
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/audit_r33_repair.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/verify_r33.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r33_repair/repair --compare r33_repair/source --label "R33 ATLAS REPAIR" --compare-label SOURCE --fps 12 --crop 848 204 265 178 --zoom 4 --out r33_repair_eyes_slow.mp4
```

Reports: `logs/r33_occupancy.json` (+ `.npz`), `logs/r33_atlas.json`,
`logs/r33_repair.json`, `logs/r33_withheld.json`, `logs/r33_verification.json`.

## 8. Starter prompt for round 34

> Stay on eyes. Read round33.md, then round32.md. Round 33 replaced round 32's
> per-frame-refit fill with ONE sequence-wide registered appearance atlas per eye plus a
> smooth temporally-smoothed illumination field. It is measurably better and **not yet
> accepted**: 29 % / 18 % better than round 32 on 5,739 / 1,599 withheld pixels across a
> genuine temporal split, at 1.36× / 1.24× the plate's texture floor, with the filled
> region now jittering LESS than the source's own pixels at median and p95. Do not
> re-derive any of that and do not go back to a per-frame fill.
> **The two things to fix are named and measured.** First, mask shape: IoU between
> adjacent frames is median 0.78 but p5 0.19 / 0.23 with a minimum of 0.0, and footprint
> size changes by up to 43 / 56 pixels — the mask still jumps outright on a minority of
> frames, and that is shape flicker. Consider predicting the footprint's position in eye
> coordinates from its own trajectory (it sweeps smoothly, 50.6 / 22.0 px of arc over
> the shot) rather than re-detecting independently each frame. Second, a light residue
> survives on 112 of 140 left eyes, median 3 px and max 27, concentrated on partly
> lidded frames where the illumination ramp mispredicts.
> Carry these forward rather than re-deriving them: the highlight SWEEPS the eye's own
> frame, so 506 of 507 ever-corrupt cells left and 550 of 621 right are observed clean
> elsewhere — observed-pixel repair is possible and the occupancy map is in
> `logs/r33_occupancy.json`; the limbus landmark jitters only 0.353 / 0.256 px rms, so
> registration must be bounded near one pixel and re-anchored each rebuild or it drifts
> while its NCC falls; an atlas may reach past rho 0.88 because it assumes no radial
> symmetry, unlike round 32's radial model; detection must never be gated on atlas
> support; and a temporal vote may only add to a frame's own mask, never erode it.
> Still true from round 32: the highlight is unclipped, the black beside it is coding
> undershoot clipped at legal black and must be left alone, and no pupil localizes.
> **Working method:** put frames in front of him constantly, inspect every frame on the
> contact sheets, and report p95 and max — never a median alone — for anything temporal.
> Two of this round's four bugs were invisible in the logs and visible in the frames.
> Preserve exact keyed fallback, the 300/304 protected checks, generic input support and
> observed identity. No key retuning, no capture requirements, no impossibility claims.
> Produce a temporal review and round34.md with measured findings and remaining physical
> work.
