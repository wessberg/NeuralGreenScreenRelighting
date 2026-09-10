# Round 36 handoff — the lash guard works, and it cannot separate lash from specular

**Date:** 2026-09-10. Starts from round 35. Production defaults unchanged.

**Delivered:** an image-derived lash guard that cuts painting on her lashes by 69-94 %
depending on variant, a fitted lid shadow, the fix for the softbox glimpses in the last
seconds, an all-frames source-vs-result atlas, and — the most useful thing in this file —
a measured proof of **why the current approach cannot satisfy both of the artist's
requirements at once**.

**Not delivered:** artist acceptance. He rejected the first r36 and the reason is in §4.

---

## 0. Read this first: the wall

The artist wants two things, both correct, both physical:

1. *"we do not draw our own pixels ON or above that eyelid arc"*
2. *"you replace less of the softbox than you did before"* — the reflection must go

**On this plate those two requirements are in direct conflict, because the lash, the
lid-shadowed upper iris and the specular's own fringe overlap in level.** Measured on
source 740 left, one column, walking down from the lid:

```
skin   .49 .47 .40
DARK   .22 .21 .21 .20 .16 .22        <- rows 11-16: lash AND shadowed iris, one run
spec   .37 .58 .82 .81 .78 .50        <- rows 17-22
```

There is no photometric boundary between "lash" and "iris" in that dark run. The lash line
therefore lands anywhere inside it — on this frame at row 19, which is **inside the
specular**, so the guard blocked the brightest part of the reflection and left the white
blob the artist rejected. Her lashes run 0.13-0.35 display, her lid-shadowed iris 0.15-0.30
and the specular's fringe 0.30-0.45. **No threshold separates them.**

Five variants were built and measured. All 300 eyes, one fixed yardstick, lash pixels =
changed pixels above the lash line, dark = those with source luma < 0.35 (i.e. actually
lash), fragments = bright blobs left in the iris of the RESULT:

| run | what it does | lash px | dark px | fragment px |
|---|---|---:|---:|---:|
| `r35_final` | no guard | 10 607 | 8 752 | **577** |
| `r36_final` | guard, +2 px offset | 4 950 | 3 818 | 1 376 |
| `r36d` | guard, bright source exempt | 1 950 | **597** | 1 327 |
| `r36e` | + the specular's black ring exempt | 3 277 | 1 924 | 1 318 |
| `r36g` | + halo exempt by local contrast, **lid shadow OFF** | — | — | — |

**Every variant trades one of his complaints for the other.** The way out is not another
threshold; it is a lash line that is geometrically right — `eyelid_fit.py` correct on the
frames where it currently sits on the lid crease. That is round 37.

## 1. What the artist said, and he was right three times

> now the softbox reflection is actually *more* visible than before ... there are more
> bright/white pixels right under the eyelid ... because you replace less of the softbox

> You tend to use your instruments, but I would wish you LOOKED instead. Just split out
> all frames of that video, isolate the eyes, and LOOK.

> 1. Your instruments didn't catch it, so your instruments need to be better.
> 2. You didn't LOOK at the frames. You MUST look at the frames.
> 3. It's still not entirely temporally stable, and the pupil and iris still shifts around

## 2. The instrument was blind, and how

`review_r36_atlas.py`'s softbox test located the specular **in the source by a threshold**
and then checked those pixels in the result. So it could only ever find what its own
threshold had already decided was the softbox. When the guard sheared off the highlight's
upper-left corner at 0.45-0.62 display it reported **no change on any frame** — 0 eyes
worse — while the artist could see a white block in the corner of her eye.

It now reads the **RESULT alone**: a leftover fragment is a bright blob against a smooth
iris, which is `pixel - median(pixel neighbourhood)` inside rho < 0.90. No reference image,
no model of the reflection, nothing shared with the repair. It reproduced his complaint
immediately: **r35 137 eyes, r36 199 eyes.**

AGENTS.md already said a metric that shares an assumption with the bug is blind to it. It
happened anyway. **The lesson is that the instrument must be derived from what the EYE
does, not from what the repair thinks it is doing.**

## 3. Looking found every one of this round's bugs

Not one of these was visible in a number, and two of them were my own "fixes":

* **The white tetris block.** At source 721 left the lash minimum is at patch row 14 and
  **the highlight starts at row 15-17**, so the fixed `+2 px` guard offset landed inside
  the specular and sheared off its upper-left corner.
* **The footprint exemption put lashes back.** The argument for it — *the footprint is
  where the plate is brighter than the canonical iris, and a lash is darker* — is FALSE
  once the lid shadow is modelled: the canonical iris near the lash is then darker than a
  lash, so the detector marks lash pixels as excess. Measured: 1333 pixels changed above
  the lash line, **70 % of them dark**.
* **The lash-inner-edge walk made it much worse.** Source 736-745 came back with a large
  white blob because the walk ran down through the merged dark run and put the ceiling
  below the specular.
* **The black crescent.** Exempting only the bright lobe of the specular left its coding
  undershoot standing — a dark ring round the top of the pupil at 736-745, which is
  exactly the artefact rounds 32-34 were rejected for.

## 4. What is in `r36e`, and its honest state

* **`lash_ceiling()`** — a per-column lash line read from her photograph, applied last so
  it outranks the fill, the removal's excursion past the fitted lid, and the footprint
  union. It can only ever remove paint.
* **`--lash-bright 0.45`** — the guard does not apply to source pixels a lash cannot
  reach, with their legal-black ringing (`--lash-ring 2.0`) exempted alongside so one lobe
  of the artefact is never deleted without the other.
* **The ceiling is temporally stabilised** by a three-frame median in the eye's own
  coordinates. Raw, it moved up to **0.8-0.9 iris radii (10 px) between adjacent frames** —
  that is the artist's *"temporal instability in the colors in the region that changes"*.
  A median of three adds no lag on smooth motion (the median of a monotone triple is the
  middle sample) while outvoting a frame whose argmin flipped a row.
* **`fit_lid_shadow()`** — `1 + A·exp(−depth/tau)` below the fitted lid margin, ONE shape
  per eye for the shot, only a scalar amplitude per frame. A = +0.44/τ=3.2 px left,
  −0.115/τ=5.2 px right. It removes about a quarter of the seam; the residual is a DIP,
  which one exponential cannot make.
* **The `ncc = 0.0` veto is fixed.** `atl.build`'s `register()` returns None when it finds
  no overlap and leaves `ncc` at its initialiser; source 843, 846 and 848 left carried
  exactly 0.0 between neighbours at 0.985/0.988/0.975 and were declined outright, leaving
  the whole softbox in frame. With the registration FROZEN every frame uses the same median
  offset, so an unregistered frame is placed as well as its neighbours. Untreated open
  eyes 3 → 0.

Verification: 150 frames, exact source outside the weight, **300 exact keyed-eye
fallbacks**, **304 protected hashes unchanged**. `production_enabled` and
`artist_accepted` false.

## 4b. The lid shadow was REJECTED ON SIGHT and is now default OFF

Round 35 §7 asked for it and this round fitted it, and looking at the face-scale panel
killed it in one frame. At A = +0.44 on the left eye it puts a **light washed band across
the top of the iris** where her source and r35 both have a dark lid shadow — so it erased
**the very thing the artist asked to preserve**:

> it's both correct to preserve a shadow from the bottom of the eyelid along the arc of
> the eye for natural depth and separation

and the bright band above the pupil is what made the pupil read as displaced and
misshapen — his *"the pupil and iris still shifts around"*. `--lid-shadow` now defaults to
**0**. The fit is still computed and reported in `removal.json`, because the measurement
in §4 is real and round 37 may want it; only its application is off.

**Why the number and the picture disagreed.** §4's depth profile says the fill is too dark
3-5 px below the lid margin, which is true *pooled over the shot*. Applying a per-eye
constant correction then over-brightens the frames that were already right, and a 20 %
lift over a band a few pixels wide reads as a halo. A correction of this kind has to be
validated at face scale on frames before it is believed, not on a pooled profile.

## 5. The tracker — reproduced, diagnosed, NOT fixed

The artist: *"there are times where she looks a bit down and our iris didn't follow along
at the exact right time"* and *"jump into position ... as it corrects the mask offset
suddenly"*.

Measured against the independent estimator in `audit_r35_limbus.py`:

| | left | right |
|---|---:|---:|
| median &#124;offset_x&#124; | **0.05 px** | 0.05 px |
| median &#124;offset_y&#124; | **0.42 px** | 0.50 px |
| worst offset_y | **−4.94 px** | −4.53 px |

**The error is almost purely VERTICAL, always negative, and every excursion is on a frame
where the lid hides the top of the iris** (15-24 edge points against a typical 41). The
geometry says why: with the radius held at the shot constant, a visible arc that is only
the sides carries almost no information about the centre's HEIGHT, so the circle slides
DOWN and snaps back when the lid opens.

`--trust-se` was added to shrink each frame's centre toward its neighbours in proportion to
`centre_se_px`, the fit's own standard error along the worst-determined axis. **It barely
fires** — 6 frames left, 21 right, max move 0.27 px — because **the fit is confidently
wrong on those frames**, the same failure mode as the eyelid fit on the lid crease. Its own
uncertainty estimate does not flag them. Default is 0.35; it does no harm and does not
solve this.

## 6. Starter prompt for round 37

> Stay on eyes. Read round36.md §0 first, then §5.
>
> **Do not build another photometric lash line and do not tune `--lash-bright`.** §0 has
> the measurement that kills that whole family: at source 740 left the lash and the
> lid-shadowed upper iris are ONE dark run at 0.16-0.22, and the specular's fringe reaches
> down to 0.30, so lash, iris and reflection overlap in level and no threshold separates
> them. Five variants are tabulated with what each one cost; every one trades the artist's
> "don't paint my lashes" against his "the softbox is still there".
>
> **The way out is a geometrically correct lash line.** `eyelid_fit.py` is audited to
> 0.43 px and is still confidently wrong where it matters — seven pixels high on the LID
> CREASE at source 727 left, abstaining entirely at 723 left. Fix the fit on those frames
> and the guard needs no photometric escape hatch at all. Round 35 §3 records that the
> crease was a known failure mode it believed it had fixed; it has not.
>
> **The same fix probably lands the tracker too** (§5): both failures are "the estimator is
> confident and wrong when the lid covers the top of the eye", and both would be helped by
> knowing where the lid actually is.
>
> **Method, and the artist said it plainly this round:** split every frame, isolate the
> eyes, and LOOK — before and after every change. Every bug this round was invisible in the
> numbers and obvious at 18x, including two of my own fixes. When a metric says a change is
> neutral and he can see it, the metric is wrong: build the instrument from what the eye
> does, not from what the repair believes.
>
> Preserve exact keyed fallback, the 300/304 protected checks, generic input support and
> observed identity. Report p95 and max, never a median alone. The scene catchlight stays
> PARKED.
