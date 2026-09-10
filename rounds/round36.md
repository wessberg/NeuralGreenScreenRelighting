# Round 36 handoff — stop painting on the lashes; the lid shadow is modelled, not solved

**Date:** 2026-09-10. Starts from round 35. Production defaults unchanged.

**Delivered:** an image-derived LASH CEILING that stops the repair drawing on her eyelid
or lashes at all, a fitted LID SHADOW so the fill carries the gradient her lid casts, the
fix for the softbox glimpses in the last seconds, and the all-frames source-vs-result
atlas the artist asked for.

**Not delivered:** artist acceptance. The seam is improved and still measurable, and a
handful of specular fringe pixels survive on squinting frames. Both are in §6.

---

## 0. What the artist said this round, because it drove everything

Round 35 ended with him rejecting `r35_eyes_final.mp4` for "the region around the upper
eyelid/eyelashes". Round 35 §7 had already measured a luminance step there and concluded
it was illumination rather than masking. **That conclusion was half right, and the half it
got wrong was the important half.** He looked at a frame and said so directly:

> considering the softbox reflection will never be present in the eyelashes or along the
> eyelid itself, I feel like we should be masking those out ... Right now it seems we're
> painting into those two for some reason? ... In the lower left image I can see how we
> draw stuff on the "arc" of the eyelid that was never there in the source

and then:

> I think it's both correct to preserve a shadow from the bottom of the eyelid along the
> arc of the eye for natural depth and separation, and it's also correct and important
> that we do not draw our own pixels ON or above that eyelid arc

and:

> in the last seconds of this video, I still see some glimpses of the original softbox
> reflection in her eyes ... make sure you have an atlas of all frames in the video that
> includes the source vs our current

**He was right on all three, and each was measurable the moment it was looked for.**

## 1. The headline number

`review_r36_atlas.py` reads only the source photograph, the result, and the written
weight map, over all 300 eyes:

| | R35 (the file he rejected) | R36 |
|---|---:|---:|
| eyes with paint ON or ABOVE the lash | **263 of 273** | **0** |
| eyes with a surviving softbox | 8 | 7 (1 px each) |
| eyes with a brightened lash/lid (`audit_r35_defects.py`) | 113, rise to 0.306 | **95, rise to 0.208** |
| untreated eyes with the eye open | 3 (843, 846, 848 left) | **0** |
| exact keyed-eye fallbacks / protected hashes | 300 / 304 | 300 / 304 |

263 of 273 is not an occasional failure, it is the normal case, and round 35 shipped it
while measuring a 0.08 illumination step and calling that the cause.

## 2. `lash_ceiling()` — the guard, and why it is not another anchor

Two mechanisms put paint on her lashes, both found by drawing the weight on the frame:

* **source 727 left** — the fitted eyelid margin sits about **seven pixels above the
  lash, on the LID CREASE**, and the fill runs right up to it;
* **source 723 left** — the eyelid fit **abstains entirely**, so there was no lash bound
  at all.

`eyelid_fit.py` is fitted to the image and audited to 0.43 px and is still confidently
wrong on these frames. **A bound that is sometimes wrong in the permissive direction
cannot be the only thing between the fill and her lashes.**

So the ceiling is a GUARD, not an anchor: it is applied last — after the fill, after the
removal's excursion past the fitted lid, after the union with the tracked footprint — and
it can only ever REMOVE paint. The worst case is a pixel of iris given up; it cannot
introduce a wrong-anchor artefact the way a new tracker could.

**The rule, per column:** walk DOWN from the lid skin at the top of the crop to the first
dark run that reaches lash level, and put the ceiling just below its darkest row. Three
things are masked out first or it finds them instead: the corneal specular, its
legal-black coding undershoot, and the pupil.

### Four wrong versions, all of which cost a pass, none visible in any log

1. **Anchored on finding skin above.** The corneal specular is **brighter than her
   eyelid** (0.78 against 0.71-0.77), so it was taken for skin and the ceiling landed in
   the middle of the iris.
2. **The column's global minimum.** Landed on the pupil's own edge, several pixels below
   the lash — the review sheet showed the ceiling stepping down beside the highlight.
3. **The first minimum walking UP from inside the eye.** Same failure at the corners,
   where her tear line and the canthal shadow are darker than any lash.
4. **The undershoot mask excluded everything under 0.22 within 5 px of the specular** —
   and at a 24-pixel iris that **is her lash**, so the walk stepped straight over the
   thing it exists to find. The real undershoot clips at legal black, 0.02-0.09. This one
   is the reason version 3 looked hopeless when it was nearly right.

**Her columns, read down at source 727 left, four pixels right of the iris centre:**

```
skin  .73 .73 .73 .71 .67 .61 .64 .64 .63    lid, then the crease
fall  .55 .42                                the lid margin
LASH  .24 .23 .19 .13 .13                    the darkest run in the column
iris  .15 .18 .19 .21 .23                    rising monotonically below it
```

Note that the lash (.13) and the shadowed upper iris (.15-.23) are **not separable by
level**. Do not try. What separates them is that the lash is a minimum reached from bright
skin above, and the eye only rises below it.

## 3. The specular clamp — BUILT FOUR WAYS, ALL REJECTED, now OFF. Read this before rebuilding it.

The guard clips the top of a softbox that sits high under a half-closed lid. Measured on
the specular core (rho < 0.6), source 736 left goes from clean at 0.26 display to a
**0.59 remnant**. So a clamp that lifts the ceiling over the specular looks obviously
right. It is not, and this cost most of the round:

| clamp definition | eyes painting above the lash |
|---|---:|
| by level, `lum > 0.45` inside the limbus | took her **LID SKIN** (0.71-0.77) and painted 0.40 over lash at 0.13 |
| compact bright blobs only | took her **WET LID MARGIN** highlight, 7 rows into the lid |
| per column rather than per blob | punched one-column holes through the fill, `w = 255 255 255 0 255 255` — a comb |
| the round-32 **footprint** (bounded, the repair's own detector) | **56 eyes**, median 24 px, max 80 |
| the footprint's whole bright **component** | **164 eyes** — on a squint the specular touches sclera and the component runs up under the lid |

The footprint version is the tempting one: it is bounded, it is the blob the repair had
already decided to remove, and it still puts a median of 24 px of paint on her lashes on
the half-closed frames (715-718, 826, 836-838) where the lid fit is least reliable.

**The artist's rule is not a tolerance.** The clamp is off, the residue is reported in §6,
and the way in for round 37 is a lash line that is right on a squint — not a licence to
paint over it.

**A measurement mistake worth recording:** the clamp was first removed on the strength of
a test that said it was worth two pixels. That test located the specular with a
sclera-relative threshold inside rho < 0.98 and **under-counted it**; measured on the
specular core the guard was clipping real highlight. A metric that shares an assumption
with the thing it measures is the recurring failure in this repo — AGENTS.md says so, and
it happened again here.

## 4. `fit_lid_shadow()` — the shadow, one shape per eye for the shot

The artist wants the shadow kept: *"it's both correct to preserve a shadow from the bottom
of the eyelid along the arc of the eye for natural depth and separation"*. Round 34 paints
the canonical iris with ONE scalar gain per channel per frame, so it cannot carry one.

The model is `S(d) = 1 + A · exp(-d / tau)` in **depth below the fitted upper lid margin**,
fitted once per eye for the whole shot on pooled observed-over-canonical ratios with each
frame normalised by its own level beyond depth 8. Bounded, monotone, two parameters,
17,000 samples per eye — it cannot do what round 33's unconstrained per-frame ramp did,
which was extrapolate 12-fold. `A` is SIGNED, and it has to be:

| | amplitude | tau | weighted rms | flat |
|---|---:|---:|---:|---:|
| left | **+0.4396** | 3.2 px | 0.0511 | 0.1063 |
| right | **−0.1150** | 5.2 px | 0.0613 | 0.0704 |

**The canonical iris is too DARK near the lash on one eye and too BRIGHT on the other.** A
model that could only darken would be fitting half the defect.

The fit is applied BEFORE the scalar gain is refit, so the scalar levels a prediction that
already bends the right way; fitting the scalar first and bending afterwards would move the
level of the whole fill every time the shape changed.

### It is a partial fix, and here is exactly how partial

Pooled over 150 frames, normalised per eye, the specular and its ring excluded, the pupil
excluded (`rho > 0.5`), left eye, result minus source at matched depth below the lid:

| depth | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| R35 | −0.152 | −0.203 | −0.125 | −0.015 | +0.002 | +0.009 |
| **R36** | **−0.090** | **−0.149** | **−0.050** | +0.035 | +0.031 | +0.027 |

About a quarter of the deviation is removed. **The residual is a DIP — too dark at 3-5,
slightly too bright at 6-8 — and a single decaying exponential cannot produce a dip.** That
is the shape round 37 needs, and the evidence for it is now scripted rather than hand-made.

## 5. The softbox glimpses in the last seconds — a placeholder read as a measurement

`ncc` on source **843, 846 and 848 left** is exactly **0.0**, with neighbours at 0.985,
0.988 and 0.975. That is not a low correlation: `atl.build`'s `register()` returns None
when it finds no candidate with enough overlap and **leaves `ncc` at its initialiser**. The
veto read the placeholder, declined the eye outright, and the whole softbox stayed in the
photograph — which is what the artist saw.

The veto made sense when the offset was per frame. **Frozen, every frame is placed by the
same median offset, so an unregistered frame is placed exactly as well as its neighbours.**
`--freeze-registration` now skips the `registered` veto; the NCC floor still applies to
frames that did register, so source 827 (ncc −0.26, a real negative correlation) is still
correctly declined. Untreated open eyes: 3 → 0.

## 6. What is still wrong

* **A few specular fringe pixels survive on squinting frames** — source 843, 845, 846, 848
  left, 5-11 px each at 0.66-0.81 display, sitting above where the lash walk puts the
  ceiling. This is the price of §3 and it is the top item for round 37. The fix is a lash
  line that is right on a half-closed eye, not a clamp.
* **The lid shadow is a quarter corrected** (§4). The residual is a dip, not a decay.
* **The seam metric is still large**: |step| p95 0.191, max 0.573 over 273 eyes. Note the
  boundary now sits AT the lash, where her own gradient is steepest, so this statistic is
  noisier than it was — treat §4's depth profile as the primary evidence, not this.
* **7 eyes keep 1 px of softbox** (700 right, 712/713/827/838/840/843 left); most are
  blink frames where the eye is closed and untreated.
* **The pupil is still a hard-edged black disc** and the canonical iris still has **no
  fibre texture**. Unchanged from round 35 §8; both are single-image problems.
* **The scene's own catchlight** stays PARKED.

## 7. New tools

| script | what it is for |
|---|---|
| `review_r36_atlas.py` | **the artist's atlas.** Every eye in the shot, source above result at eye scale, with SOFTBOX / LASH / STEP burned onto each tile and a coloured border, so a bad frame is found by scrolling rather than by reading 300 numbers. Writes `logs/r36_atlas_<name>.json`. |
| `review_r36_lash.py` | the lash ceiling drawn in green on the photograph, beside the fitted lid margin in yellow, so the guard is judged against her lashes rather than against another estimator. |
| `audit_r36_seam.py` | round 35 §7's hand measurement, as a script. Reads only source, result and the written weight. |

## 8. Mistakes made this round

* **Round 35's §7 diagnosis was believed for too long.** It measured a real illumination
  step and concluded "it is not masking". It was both, and masking was the larger half by
  far — 263 eyes against a 0.08 display step. The artist found it by looking at one frame.
* **The specular clamp was built four times** (§3), and the version that removed it was
  justified by a metric that under-counted the specular.
* **The acceptance metric shares code with the thing it measures.** `review_r36_atlas.py`
  calls `lash_ceiling()` for its LASH test, so changing the ceiling changes the yardstick
  and the LASH counts are not comparable across runs where that function changed. It is
  still the right question; round 37 should give it an independent lash reference.

## 9. Reproduce

```powershell
$P = "dependencies/CorridorKey/.venv/Scripts/python.exe"
$S = "music_videos/reporter_test/shots/reporter_a"

& $P -B scripts/review_r36_lash.py --shot $S --side left      # the guard, on frames
& $P -B scripts/eye_soft_removal.py --shot $S --name r36_final `
     --atlas-frames "$S/composite/r34_gen_wide/inpaint" --atlas-mode replace `
     --fill-iris 0.99 --grain 0 --pupil-target 0.34
& $P -B scripts/verify_r34.py         --shot $S --name r36_final
& $P -B scripts/audit_r35_defects.py  --shot $S --name r36_final
& $P -B scripts/review_r36_atlas.py   --shot $S --name r36_final   # all 300 eyes
& $P -B scripts/audit_r36_seam.py     --shot $S --name r36_final

& $P -B scripts/review_r35_eyes.py --shot $S --name r36_eyes_final `
     --dirs "r36_final/source,r35_final/removal,r36_final/removal" `
     --labels "SOURCE|R35 (what you watched)|R36"
```

Verification for `r36_final`: 150 frames, **exact source outside the reported weight**,
**300 exact keyed-eye fallbacks**, **304 protected hashes unchanged**.
`production_enabled` and `artist_accepted` are false.

## 10. Starter prompt for round 37

> Stay on eyes. Read round36.md §3 first, then §4, then §2.
>
> **Do not rebuild the specular clamp.** Five versions are recorded in §3 with the number
> of eyes each one painted above the lash; the best bounded one still put a median of 24 px
> on her lashes. The artist's rule is settled and it is physical: a corneal reflection is
> never on a lash, so nothing may be drawn there. What is actually needed is a **lash line
> that is right on a HALF-CLOSED eye** — that is where the remaining specular fringe
> survives (source 843, 845, 846, 848 left, 5-11 px at 0.66-0.81) and where the lid fit is
> least reliable. Fix the line, and the residue goes without any clamp.
>
> Then the **lid shadow**, which §4 corrects by about a quarter. The residual is a DIP —
> too dark 3-5 px below the lid margin, slightly too bright at 6-8 — and a single decaying
> exponential cannot make a dip. Keep the round-34 construction: one shape per eye for the
> shot, only an amplitude moving per frame. Do not go back to a per-frame ramp.
>
> `review_r36_atlas.py` is the artist's own acceptance tool and he asked for it by name —
> run it before and after every change and look at the sheets, not just the summary line.
> Give its LASH test an independent lash reference first: it currently calls
> `lash_ceiling()`, so it shares code with what it grades (§8).
>
> **All three eye anchors stay as they are** — `limbus_fit.py`, `eyelid_fit.py`, and the
> frozen atlas registration. Do not put a mediapipe contour back. Preserve exact keyed
> fallback, the 300/304 protected checks, generic input support and observed identity.
> Look at frames BEFORE and AFTER every change; every one of this round's five wrong
> versions was invisible in the logs and obvious in the pixels. Report p95 and max, never
> a median alone. The scene catchlight stays PARKED.
