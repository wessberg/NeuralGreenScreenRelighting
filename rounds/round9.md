# Round 9 handoff — better iris correspondence, no supported catchlight repair

**Date:** 2026-09-08. Starts from round 8 and the round-4-equivalent production defaults.
**No visual fix for the original catchlights, crown sheen or lip gloss was delivered.**
The new correspondence experiment improves recovery of withheld, observed pixels, but has
zero confident support at the detected reflections themselves. No photorealism row becomes
green. Artist verdict remains pending.

## 1. Delivered and verified

`scripts/audit_r9_iris.py --shot <shot>` runs a reporter-only correspondence diagnostic on
all 150 keyed source frames. It neither invokes nor changes the production compositor.
It writes a hidden-pixel comparison video, correspondence evidence and independent checks.

The experiment starts with round 8's 41x41 iris-centre/radius grids, then fits a residual
affine transform between target and donor using only visible, non-bright context. The
OpenCV implementation supports masks on **both** images; Gaussian filtering is set to 1
to avoid smoothing hidden target values into the fit. Mapping direction follows the
[OpenCV 4.13 masked ECC documentation](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html)
and is independently tested on known translations in four directions.

This is a bounded image registration model, **not a physical cornea/iris model**. Accepted
affine matrices have singular values between 0.90 and 1.10 and move the four interior grid
corners no more than four grid pixels. It cannot explicitly model pupil dilation, refraction,
occlusion or motion blur. Enlarging an approximately 24-pixel-wide iris adds no captured detail.

The new method is isolated. SHA-256 hashes before/after match for **302 protected files**:
150 keyed foregrounds, 150 baseline `composite/ratio` PNGs, `shot.json` and the production
compositor. Hair and lips receive no new treatment. No new Cycles render, keying, normals,
albedo, source-light fit, removal pass or replacement catchlight is claimed.

## 2. Stronger hidden-pixel control and measured correspondence improvement

Round 8 hid values on an interpolated grid. Adjacent grid samples can share the same native
source pixels, so their RGB is not independent evidence. Round 9 computes all four native
bilinear contributors to each hidden sample, then excludes **every target context sample
sharing any contributor**. This exclusion is defined from sampling geometry, not RGB.
It does not retroactively alter round 8's historical report or prove its entire error arose
from this overlap.

Four fixed radius-three grid discs remain the evaluation locations. A test is eligible if
the old aperture mask contains the whole disc and the old expanded candidate-glint mask
does not touch it. There are **470 left-eye and 318 right-eye eligible patches**. These are
observed non-candidate pixels, not certified diffuse iris texture. The masks and cached
landmarks can still be wrong near the lower rim and lids.

For prediction, all hidden target/overlap RGB is scrubbed before brightness statistics,
shortlisting, alignment, ranking and confidence. The new conservative exclusion rejects
all sufficiently bright pixels, including components touching boundaries. It is an exclusion
for matching, **not a validated reflection-removal mask**. Warped donor pixels are used only
when every interpolation contributor passes its donor mask.

Both methods use the same exclusions and source-footprint guard. Twelve donors are
shortlisted by observed-context error, excluding the target and its +/-3 adjacent frames.
The best seven usable fits contribute a per-pixel median with at least three distinct
donors. Donors may be adjacent to one another; the temporal guard is against the target.
The control leaves the initial centre/scale registration fixed. The experiment fits the
bounded residual affine transform. There is no photometric gain or colour fit.

**Compare the same pixels**, because alignment abstains more often. These results are on
the intersection of the two methods' support, with equal weight per supported test patch:

| Withheld control | Left fixed | Left aligned | Right fixed | Right aligned |
|---|---:|---:|---:|---:|
| Matched supported patches | 375 | 375 | 225 | 225 |
| Matched grid samples | 9,711 | 9,711 | 6,352 | 6,352 |
| Median relative RGB absolute error | 11.67% | 7.27% | 10.45% | 6.09% |
| p95 relative RGB absolute error | 30.76% | 16.72% | 35.14% | 15.03% |
| Median display-channel MAE, 0–255 | 2.65 | 1.81 | 3.28 | 2.04 |
| p95 display-channel MAE | 7.34 | 3.18 | 7.97 | 3.98 |
| Patches with display MAE over 5 | 50 | 2 | 48 | 6 |

Alignment supports some pixels in **375/470 left and 225/318 right eligible patches**,
compared with **470 and 317** for the fixed control. Partial patches count as supported;
the report retains actual pixel counts. These numbers must not be presented as universal
recovery or compared directly with round 8's different masks as a pure alignment ablation.

For the fixed lower test hole, compare adjacent-frame changes in reconstruction residual,
after subtracting the actual photographed change. Both methods use identical pixels and
adjacent pairs: **101 left, 89 right**. Median residual-change MAE falls from **2.73 to 1.76**
display levels left and **2.35 to 1.75** right. Corresponding p95 values are **6.00 to 2.83**
and **5.77 to 3.28**. This is conditional temporal evidence on a known patch, still including
iris-grid tracking error; it is not a flicker measurement on repaired reflections.

## 3. The important failure: no confident evidence beneath the original reflections

Before evaluating the sequence, confidence was defined as median donor-context error at
most **2 display levels**, plus per-pixel donor median absolute deviation at most **2 levels**.
This is a diagnostic gate, not a calibrated probability or an artist acceptance threshold.

It accepts **2,875 left-eye and 1,258 right-eye hidden test samples**, spread over 113 and 46
patches. Their median display MAE is **1.65 and 1.87**, with p95 **2.37 and 2.61**. No accepted
test patch exceeds 5 display levels of mean error. Error still exists; agreement is not
proof of correct texture, and these results are conditional on eligibility and abstention.

Apply the same predictor to the old detector's actual compact reflection candidates,
without displaying or writing a proposed repair:

| Candidate evidence over the sequence | Left eye | Right eye |
|---|---:|---:|
| Frames containing old glint candidates | 59 | 108 |
| Candidate grid samples, summed over frames | 2,601 | 4,473 |
| Samples with three usable aligned donors | 4 | 8 |
| Samples passing the confidence gate | **0** | **0** |

This does not fail merely because every registration was rejected: **53/59 left and
97/108 right candidate frames have at least three usable context fits**. Their fitted
donors almost never provide three unmasked observations at the candidate location itself.
The remaining 6 and 11 frames lack three usable fits.

**Do not lower these exclusions just to manufacture coverage.** Round 8's apparent lower
observations did not establish correspondence or clean texture. Round 9's near-zero support
is conditional on conservative masks, the short clip, affine bounds and shortlisting; it
does not prove every temporal method or longer source interval impossible. It does establish
that this implementation cannot safely remove the source catchlights on this sequence.

The original removal prerequisite failed, so no new corneal pass or lid-only compositing
patch was promoted. **Original-reflection removal and actual lid/brow/lash source occlusion
are still jointly required** before replacement catchlights can enter production.

## 4. Review artefact and actual inspection

Under `music_videos/reporter_test/shots/reporter_a/final/`:

- **`r9_iris_withheld_comparison.mp4`** — 150 frames, 60000/1001 fps, CRF 12; top row
  subject-left eye, bottom row subject-right. Columns are photographed truth, fixed donors,
  aligned donors. Only the predetermined lower hidden control patch is substituted.
  Purple denotes unavailable/unsupported test pixels, not a rendered catchlight.

The full video decodes to **150 frames** at the correct source rate. Every encoded frame was
visually inspected in **ten consecutive 15-frame sheets**, each frame reduced to 492x360.
Native enlarged comparison PNGs **67 and 120** were also inspected. This was ordered-frame
inspection, **not real-time playback and not artist approval**.

Observed: at frame 67 the fixed predictor inserts a dark patch near the lower iris rim;
alignment abstains there. Blinks and some reopening frames lose support. Supported lower
iris patches are often less conspicuous with alignment, but the original softbox reflection
remains plainly present throughout the visible-eye portions of this diagnostic. It is not
a proposed improved composite.

Structured evidence:

- `logs/r9_iris_audit.json`: matched/unmatched errors, coverage, per-patch donors and scores,
  confidence tests, candidate failures, temporal residuals, code checks and inspection record.
- `composite/r9_iris/correspondence_detail.json`: regenerable full affine matrices and report.
- `composite/r9_iris/holdout_review/`: all 150 comparison PNGs.
- `composite/r9_iris/review_*.jpg`: ten ordered video-decode sheets.
- The video's JSON sidecar records encode provenance; the shared `review_video.json` now
  describes this review. Derived caches and video follow the existing gitignore policy.

Reproduce in the pipeline venv:

```powershell
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/audit_r9_iris.py --shot music_videos/reporter_test/shots/reporter_a
```

The numerical run intentionally writes inspection as pending; a fresh encode does not
inherit a claim to have been visually inspected. This session's completed inspection was
recorded afterward. `--checks-only` runs the synthetic checks without regenerating the audit.

Checks pass: four known-motion directions (centre error **0.0061–0.0258 grid pixels**),
native-image perturbation outside the footprint guard (bit-identical context), ten real
hidden-RGB perturbation checks including supported aligned predictions, missing-evidence
abstention, full video decode and protected-file hashes. These are implementation checks;
**the actual reflection-recovery experiment does not pass**.

## 5. Settled and housekeeping

Keep round-4-equivalent defaults: rejected separation and catchlight replacement stay off.
No new on-set reference, generative performer, key-threshold change or persistent Cycles data.
All three original visual complaints remain unresolved. Historical rounds/reports are preserved.

`AGENTS.md` now sorts round filenames **numerically**, so round 10 will not disappear behind
round 9 in the "read latest" command. Its reflection note records round 9 without declaring
a repair. No change to camera/capture configuration was made.

## 6. Starter prompt for round 10

> Read round9.md and the rejected-stage findings in rounds 8, 7 and 6. Masked affine
> correspondence improves hidden known-pixel recovery but gives zero confident support
> beneath the detected original reflections. Do not turn that metric improvement into a
> removal mask or a delivery claim. Before another eye-repair attempt, determine whether
> a wider interval of the existing original footage supplies genuinely different, matched
> observations beneath the reflection, and audit the candidate masks against the images.
> Preserve native-footprint exclusion and matched-support controls; report missing coverage.
> If clean evidence remains unavailable, state that limitation rather than inventing iris
> texture. Replacement requires source removal plus actual lid/brow/lash occlusion together.
> Hair/lips remain open and need material-aware temporal evidence, not the rejected opening
> subtraction. Keep production defaults, make a complete comparison video, inspect the
> sequence and write round10.md with a starter prompt for round 11.
