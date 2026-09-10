# Round 35 handoff — both eye trackers now fit the IMAGE; the lid shadow is the seam

**Date:** 2026-09-10. Starts from round 34. Production defaults unchanged.
**Delivered:** the limbus fitted to the image edge, the eyelid fitted to the image edge,
the last per-frame refit in the eye path frozen, the canonical iris stopped being poisoned
by its own mask, and an all-frames acceptance test for the two defects the artist names.
**Not delivered:** artist acceptance. His verdict on the final file:

> there are still problems in the region around the upper eyelid/eyelashes … it still
> doesn't look seamless

That is measured in §7 and it is **not** a masking problem. Read §7 first if you are short
of time.

---

## 1. What round 34 said was blocking, and what actually was

Round 34 measured the mediapipe four-point iris ellipse at 0.936 of the observed iris
radius on the left eye and called it the blocker. It was, and it is fixed. But it was the
first of **three** bad anchors, and the artist found the other two by eye before any metric
here did:

| anchor | error against the image | fixed by |
|---|---|---|
| iris ellipse (mediapipe 4-point) | 3.08 px from her iris centre (left), 2.59 (right) | `limbus_fit.py` |
| eyelid contour (mediapipe polygon) | sits **8.34 px inside** the true opening (left), 1.01 px outside (right) | `eyelid_fit.py` |
| atlas registration (per-frame NCC) | translated the painted iris up to **1.62 px between adjacent frames** | frozen to one offset per eye |

**The lesson is the same one round 34 wrote and this round had to learn twice more:
anything refit per frame moves, and any mediapipe contour on this plate is a guess that
has to be checked against the photograph before it is trusted.**

## 2. `limbus_fit.py` — the limbus, fitted to the image edge

Daugman's integro-differential operator finds the circle globally (a mean of the radial
derivative around the contour, so no single lash or vessel can pull it), then per-angle
sub-pixel edges and a robust geometric fit. **One radius for the whole shot** — her iris is
a rigid disc — which removes the radius/centre degeneracy that made the free fit collapse
to 8.8 px on half-closed frames. Centres are interpolated across gaps of ≤3 frames.

**No temporal smoothing, and that is a measurement.** The fitted centre's residual against
a local quadratic through its own neighbours is **0.048 px median, 0.229 p95**. There is
nothing left to filter, and filtering costs: a ±3-frame local-linear smoother moved the
centre **3.4 px** at the fast head move around source 757, which is lag — precisely the
artist's "delayed, slow to react". `--smooth-half` defaults to 0.

**Round 33 was not wrong that the landmark was smooth** (0.049 px against the same
control). Its error was a *systematic* 2.5 px offset and a 4 % size error, and no temporal
filter addresses either.

### Round 34's own audit, unchanged, on the new anchor

| | R34 landmark | R35 fit |
|---|---:|---:|
| left observed/tracked radius | 0.936 | **0.997** |
| left spread | 1.66 px | **0.24 px** |
| left p95 frame-to-frame | 2.49 px | **0.38 px** |
| right spread | 1.42 px | **0.22 px** |
| right p95 frame-to-frame | 2.19 px | **0.28 px** |

### An independent audit, sharing nothing with the fitter

`audit_r35_limbus.py` locates the limbus by a **level crossing on the max channel**,
bilinear on the image, and reads the displacement off the first harmonic of r(θ). No
derivative, no shared feature, no shared sampling, no shared model.

| offset from her own iris centre | left | right |
|---|---:|---:|
| mediapipe landmark | 3.078 px (p95 6.25, max 11.7) | 2.591 px (p95 4.71) |
| **R35 fit** | **0.422 px** (p95 1.92, max 6.92) | **0.504 px** (p95 1.90) |
| fit better, paired frames | 119 / 120 | 123 / 125 |

The anchor's *error* also stops breathing: its frame-to-frame change falls from 0.527 px
median / 10.02 max (landmark) to 0.200 / 2.67 (fit).

**Synthetic control**, a known disc at 8 bit with the plate's blur and noise: fitter centre
error 0.011 px, radius error −0.106 px; audit centre error 0.007 px, radius error −0.052 px.
Both estimators are unbiased at a tenth of a pixel before either judges the other.

### The artist's round-23 question, answered with numbers

He asked why `composite/r23_geometry/contact_024.jpg` looked perfect. Two reasons, and he
was right on the substance:

1. **Scale.** Those tiles are 70 px crops shown at about 1.7× on the sheet. A 2.5 px error
   is four sheet-pixels there and forty at the 16× he was shown this round.
2. **It was a different estimator, and it genuinely was better.** The cyan ring came from
   EllSeg/RITnet's limbus boundary through an optical fit. **That anchor was never carried
   into the fill** — rounds 32–34 used the mediapipe ellipse instead.

| offset from her iris, same independent test | left | right |
|---|---:|---:|
| mediapipe (what R32–34 used) | 3.08 px | 2.59 px |
| EllSeg boundary (round 23's ring) | 1.18 px | 0.46 px |
| R35 image fit | **0.42 px** | 0.50 px |

EllSeg is 2.6× better than what the fill had been using and was left on the floor. It runs
5 % oversize (median radius 12.90 vs the image's 12.30) and has a worse tail (12.5 px max),
which is why the new fit is preferred — but **do not repeat the mistake of leaving a better
measurement unused because it lives in another script.**

## 3. `eyelid_fit.py` — the eyelid, fitted to the image edge

The artist's diagnosis, and it was exactly right: *"It really is just the eyelid you need
to mask out. You track the image perfectly, but don't fully exclude the lid from it."*

Per column, the eye ends at the **first** bright-to-dark step walking outward from the eye
centre; a robust parabola per lid, weighted by step strength, with the **curvature held as
one number per eye for the shot** (a lid's shape is anatomy, its height is motion).

Column residual against the image: **upper 0.435 px (left) / 0.429 (right), lower 0.302 /
0.227.** 116 of 150 frames fit per eye; the rest keep the tracked polygon and say so.

**Three bugs, all found in frames and invisible in the residual:**

* Columns ran past the eye **corners** into skin and dragged the parabola into a diagonal
  sweep. The corners are the one part of the mediapipe contour that is right — sharp,
  unoccluded, nowhere near a lash — so they now bound where to look.
* Taking the **strongest** step outward found the brow and the lid crease. The lash is the
  **first** step, whatever its contrast.
* **The corneal highlight is the brightest thing in the eye**, so the first big fall above
  it is its own upper edge. Fitted without excluding it, the upper lid landed just above
  the softbox, across the middle of the iris — *the repair's own tracker reproducing the
  error the repair exists to fix.*

## 4. The registration, frozen — this was the visible jitter

The atlas registration was still refit per frame when rendering the canonical iris back.
Building the atlas *wants* that; rendering does not. With round 34's anchor the offset was
doing real work (the anchor was 3 px out and moving); with a 0.42 px anchor what remained
was NCC chasing the lid shadow, and it **translated the whole painted iris a median 0.21 px
and up to 1.62 px between adjacent frames**. That is the artist's *"our iris copy sometimes
shifts a little bit around in-between frames"*, measured.

Frozen to one offset per eye for the shot. Measured step is now **0.000 px**. The median
offset came out exactly (0.0, 0.0), which is independent evidence the new anchor is
unbiased against the atlas.

## 5. The canonical iris was poisoned by the mask that paints it

The artist rejected a warm arc following the top of the mask and named the real question:
*"why there even was this bright area around the arc … it's not present in the source"*.

The canonical iris is a per-cell median over ~140 registered frames. **It was being pooled
through the same eye-opening mask it is painted through.** Every cell near the lid took its
votes from whatever the lid fit let in on the frames that reached it — and where the fit
ran a pixel high, that is lash and lid skin. The median does not rescue it: if the fit is
high on most contributing frames, most votes are skin.

`--atlas-lid-inset 3.0` pools the canonical iris from **3 px inside** the opening it is
painted through; cells left empty are interpolated from clean iris by the hole-filling
stage that already existed for that purpose.

**Two things were tried first and both were wrong, and the artist called it:** capping the
canonical iris to her own observed radiance (`--atlas-cap`, now default 0) and a per-pixel
"the fill may not brighten" clamp. Both limited how bad the artefact looked. Neither
touched the cause. *Do not treat a symptom in the render when the canonical image is wrong.*

## 6. `audit_r35_defects.py` — the acceptance test, on all 300 eyes

His instruction: *"Look at all frames, and fix all of those where we either see a brighter
lid or part of the softbox reflection."* Both tests read **only the source photograph and
the result** — no mask from the repair, so neither can be blind where the repair is wrong.

| | R34 | R35 final |
|---|---:|---:|
| eyes with softbox left | 12 (up to 6 px) | **4 (1 px each)** |
| eyes with a brightened lash/lid | 136, rise to **0.306** | 113, rise to 0.247 |
| untreated eyes still carrying a specular | 0 | 0 |

**The audit's own false positive, recorded because it cost an hour:** 30 eyes were flagged
as surviving softbox that were her **tear film at the inner corner** — the disc reached
onto the sclera at rho 1.05. The corneal specular is on the cornea; the corner is not.
Confining the disc to 0.98 took the real count to 4.

`--lid-inset` is at a measured crossover, not a dial:

| `--lid-inset` | brightened-lid eyes | softbox left | treated eyes |
|---:|---:|---:|---:|
| 1.0 | 140 | 4 × 1 px | 271 |
| **2.0 (default)** | **113** | **4 × 1 px** | 268 |
| 3.0 | 81 | 7 × up to 4 px | 261 |

Past 2 px the fill buys lash safety with softbox.

Two more fixes this round came out of that list: the **720-right pop** (an NCC floor that
only mattered when the registration was per-frame vetoed one frame sitting between two at
0.898 and 0.942), and the canonical eye's grid, which **stopped at the limbus** — 92.5 % of
the specular pixels still standing sat at rho > 0.95 with *zero* removal weight, median rho
1.022, on the limbal ring where there was no canonical material at all. `atl.configure()`
now reaches 1.25 radii and samples to 1.10; round 32's `RHO_MAX = 0.88` was a constant for
its radial profile model, never a statement about how far her iris goes.

## 7. **What is still wrong, and it is NOT the mask — start here**

The artist's remaining verdict is the upper eyelid and lashes, and the number says the
cause is illumination, not geometry.

Measured across 258 treated eyes, the luminance step across the fill boundary in the
**upper half** of the iris — median just inside the weight, minus median just outside it:

| | step (display luma) |
|---|---:|
| in her SOURCE photograph | **+0.0824** (p95 +0.1255) |
| in the R35 RESULT | **−0.0039** (p95 +0.0255) |
| what the repair changed it by | **−0.0941** median, max +0.0451 |

**Her iris darkens toward the lash and the repair flattens that gradient.** The canonical
iris is painted with ONE scalar gain per channel per frame (`local_gain`, round 34), so it
cannot carry a shadow that varies across the disc. Where the fill meets her own darker,
lid-shadowed iris there is therefore a systematic step of about a tenth of a stop, right at
the lash line, on essentially every treated frame. **No adjustment to the eyelid mask can
remove it** — moving the boundary only moves where the step lands. That is what "not
seamless around the upper eyelid" is.

Round 33 had a per-channel linear ramp for exactly this and round 34 replaced it with the
scalar because the ramp was fitted on clean iris and then evaluated INSIDE the footprint,
where it had no evidence, with a 12-fold clamp. Both are wrong for a whole-iris fill: a
scalar cannot follow the shadow, and an unconstrained ramp extrapolates.

**The parameterisation that is now available and was not before: distance below the fitted
upper lid margin.** The lid shadow is cast by the lid, so its natural coordinate is depth
under the lid edge — and round 35 measures that edge to 0.43 px. Fit the ratio
(observed clean iris ÷ canonical iris) as a smooth, monotone, **bounded** function of that
depth, on the clean iris only, and it is interpolation rather than extrapolation
everywhere the fill needs it. One function per eye for the shot, with only its amplitude
per frame, would keep the round-34 construction intact: build the invariant once, let one
scalar move.

## 8. Also still open

* **Source 719 left** abstains on the "a repair must darken the eye" guard while both
  neighbours repair — one frame pops. The guard is right in principle; that frame needs
  looking at rather than the guard needs loosening.
* **The pupil is a hard-edged black disc** and the canonical iris has **no fibre texture**.
  Both are now single-image problems: the eye is built once, so they are tractable in a way
  they were not before round 34.
* **The scene's own catchlight** (`render_eye_catchlight.py`) is built and validated and
  the artist PARKED it. Do not switch it on unprompted — but the face-scale frames still
  say removal alone reads as a dead eye, so raise it once the lash region is clean.
* 113 eyes carry a few pixels of lightened lash, median 3 px, p95 39, worst 67 px at 0.247,
  concentrated on the blink-edge frames (705, 712, 716, 717, 723, 726, 727, 826, 834, 836,
  837, 846, 847) where the lid fit is least reliable. Some of this is §7 and will go with
  it; the rest is lid-fit quality on half-closed eyes.

## 9. Mistakes made this round, so they are not repeated

* **A video was sent before its frames were inspected**, and the artist found a lash smear
  I had introduced in that very run. AGENTS.md already says to look at frames before and
  after every change. It is not optional and it cost this round two cycles.
* **Three workarounds were tried before the cause was found** for the warm arc: a fixed lid
  margin, a brightness cap on the canonical iris, and a per-pixel brightening clamp. The
  artist rejected each as a workaround and he was right each time. §5 is the cause.
* **Two script bugs**, both silent: a stale `patch` variable read from the previous loop
  (shape mismatch, caught only by an exception), and `--lid-inset` applied as an offset to
  a *matte threshold* instead of in pixels, which made it a no-op at any value above ~0.5.

## 10. Reproduce

```powershell
$P = "dependencies/CorridorKey/.venv/Scripts/python.exe"
$S = "music_videos/reporter_test/shots/reporter_a"

& $P -B scripts/limbus_fit.py  --shot $S            # -> relight/limbus_track.json
& $P -B scripts/eyelid_fit.py  --shot $S            # -> relight/eyelid_track.json
& $P -B scripts/audit_r35_limbus.py --shot $S       # independent accuracy + synthetic control
& $P -B scripts/review_r35_track.py --shot $S       # the anchor on every frame, 10x

& $P -B scripts/eye_soft_removal.py --shot $S --name r35_final `
     --atlas-frames "$S/composite/r34_gen_wide/inpaint" --atlas-mode replace `
     --fill-iris 0.99 --grain 0 --pupil-target 0.34
& $P -B scripts/verify_r34.py --shot $S --name r35_final
& $P -B scripts/audit_r35_defects.py --shot $S --name r35_final

& $P -B scripts/review_r35_eyes.py --shot $S --name r35_eyes_final `
     --dirs "r35_final/source,r34_newiris2/removal,r35_final/removal" `
     --labels "SOURCE|R34|R35"
```

Verification for `r35_final`: 150 frames, **exact source outside the reported weight**,
**300 exact keyed-eye fallbacks**, **304 protected hashes unchanged**, all 268 treated eyes
darker than source. `production_enabled` and `artist_accepted` are false.

Reviews (display-sRGB CRF 12, review only, never a master):
`final/r35_eyes_final.mp4` (SOURCE / R34 / R35 at 10×, every frame),
`final/r35_panel_final.mp4` (the same three at face scale),
`final/r35_track.mp4` (the anchor itself, mediapipe against the fit),
`composite/r35_final/defects.jpg` (the 24 worst remaining, source beside result).

## 11. Starter prompt for round 36

> Stay on eyes. Read round35.md §7 first, then §5, then round34.md.
>
> **The artist's open complaint is the upper eyelid and lashes, and §7 measures the cause:
> it is illumination, not masking.** Her iris darkens toward the lash by **+0.082 display**
> across the fill boundary in her own photograph, and the repair flattens that to −0.004 —
> a systematic step of about a tenth of a stop at the lash line on essentially every
> treated frame, because the canonical iris is painted with ONE scalar gain per frame.
> **Moving the eyelid mask cannot fix this; it only moves where the step lands.** Do not
> spend the round on the mask.
>
> The parameterisation to try is the one round 35 made available: **depth below the fitted
> upper lid margin**, which is measured to 0.43 px. Fit (observed clean iris ÷ canonical
> iris) as a smooth, bounded, monotone function of that depth on clean iris only, so it
> interpolates rather than extrapolates where the fill needs it. Keep round 34's
> construction: build the shape ONCE per eye for the shot and let only an amplitude move
> per frame. Round 33's unconstrained linear ramp with a 12-fold clamp is the failure to
> avoid; round 34's scalar is the limitation to remove.
>
> **All three eye anchors are now fitted to the image and audited — do not re-derive them,
> and do not put a mediapipe contour back.** `limbus_fit.py` and `eyelid_fit.py` are the
> anchors; `audit_r35_limbus.py` and `audit_r35_defects.py` are the tests, and both read
> only the photograph and the result. Run `audit_r35_defects.py` before and after every
> change: it is the artist's own acceptance criterion (a brighter lid, or any softbox
> left) over all 300 eyes rather than the ones that happen to be on screen.
>
> Then the pupil, which is still a hard-edged black disc, and the canonical iris, which
> still has no fibre texture. Both are single-image problems now.
>
> **Working method, and round 35 broke it:** look at frames BEFORE and AFTER every change,
> and never send a review whose frames you have not inspected — the artist caught a lash
> smear I had introduced in the run I sent him. When he says a fix is a workaround, stop
> and find the cause; he was right three times this round. Report p95 and max, never a
> median alone. Preserve exact keyed fallback, the 300/304 protected checks, generic input
> support and observed identity. The scene catchlight stays PARKED until he says otherwise.
> No key retuning, no capture requirements, no impossibility claims.
