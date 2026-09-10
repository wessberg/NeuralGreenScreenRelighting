# Round 36 handoff — REJECTED. Read §0 and §1 before touching anything.

**Date:** 2026-09-10. Starts from round 35. Production defaults unchanged.
**Artist verdict: rejected.** His closing words:

> Apparently you can't see it. Hand off the round, I'll tackle it fresh in the next one.

That sentence is the most important thing in this file and §1 is about it.

---

## 0. The wall, measured

The artist wants two things. Both are correct and both are physical:

1. *"we do not draw our own pixels ON or above that eyelid arc"*
2. *"you replace less of the softbox than you did before"* — the reflection must go

**On this plate they are in direct conflict, because lash, lid-shadowed iris and the
specular's own falloff overlap in level.** Source 740 left, one column, read down from the
lid:

```
skin   .49 .47 .40
DARK   .22 .21 .21 .20 .16 .22     <- rows 11-16: lash AND shadowed iris, ONE run
spec   .37 .58 .82 .81 .78 .50     <- rows 17-22
```

Her lashes run **0.13-0.35**, her lid-shadowed iris **0.15-0.30**, the specular's falloff
**0.30-0.45**. There is no photometric boundary anywhere in that. A lash line built from
level therefore lands anywhere inside the dark run — on this frame at row 19, *inside the
specular*, which is why the guard blocked the brightest part of the reflection and left the
white block the artist saw.

**Six variants were built. Every one trades one of his complaints for the other.** All 300
eyes, one fixed yardstick: `lash px` = pixels changed above the lash line, `dark px` =
those whose source luma is below 0.35 (i.e. actually lash), `frag px` = bright blobs left
in the iris of the RESULT (local contrast, reads the result only).

| run | what it does | lash px | dark px | frag px |
|---|---|---:|---:|---:|
| `r35_final` | no guard — **what he watched before** | 10 607 | 8 752 | **577** |
| `r36_final` | guard, fixed +2 px offset | 4 950 | 3 818 | 1 376 |
| `r36d` | + bright source exempt | 1 950 | **597** | 1 327 |
| `r36e` | + the specular's black ring exempt | 3 277 | 1 924 | 1 318 |
| `r36f` | + lid shadow OFF | 3 240 | 1 879 | 1 318 |
| `r36g` | + halo exempt by local contrast | 3 437 | 1 879 | **972** |

`r36g` is the last state and the one on disk. It cuts lash painting by 68 % against r35 and
still leaves 68 % more bright fragments than r35. **Neither end of the table is
acceptable to him.** Do not build a seventh variant.

## 1. The failure that matters: I could not see the defect

The artist said it three times, escalating, and he was right every time:

> You tend to use your instruments, but I would wish you LOOKED instead. Just split out
> all frames of that video, isolate the eyes, and LOOK.

> 1. Your instruments didn't catch it, so your instruments need to be better.
> 2. You didn't LOOK at the frames. You MUST look at the frames.
> 3. It's still not entirely temporally stable, and the pupil and iris still shifts around

> Apparently you can't see it.

**This is a calibration warning for round 37, and it should be trusted over anything else
in this file.** Round 36 repeatedly declared frames "clean" that he could see were not —
including after being told to look, and including at 18x zoom. Concretely:

* I called `r36_final` frames 23 and 36 "clean — lash and lid fully preserved, natural
  shadow", and shipped a video. He found a white block in the corner of her eye in the
  same run.
* I called `r36f` "the bright band is gone, the pupil reads round and centred again", then
  found grey specks in the pupil one frame later, and after fixing those he still says the
  defect is there.

**So: do not take round 36's visual assessments as evidence of anything.** Re-derive them.
Where this file says a frame looks acceptable, verify it before building on it. The
numbers in §0 are reproducible and can be trusted; the words "looks clean" cannot.

A concrete method that works: put the previous result and the new one in the **SAME tile**,
one above the other, at high zoom, and step every frame. Differences that are invisible in
isolation are obvious in comparison — a frame judged alone reads as fine, and stacked under
r35 it shows a white block in the corner of her eye at a glance. That is now
**`scripts/review_r36_ab.py`**, promoted out of the scratchpad because it is the only tool
this round that found anything.

## 2. The instrument was blind, and how

`review_r36_atlas.py`'s softbox test located the specular **in the source by a threshold**
and then checked those pixels in the result. It could only find what its own threshold had
already decided was the softbox, so when the guard sheared off the highlight's upper-left
corner at 0.45-0.62 display it reported **zero eyes worse on every frame** while the artist
could see a white block.

It now reads the **RESULT alone**: a leftover fragment is a bright blob against a smooth
iris, `pixel − median(neighbourhood)` inside rho < 0.90. No reference image, no model of
the reflection, nothing shared with the repair. It reproduced his complaint at once —
r35 137 eyes, r36 199.

AGENTS.md already said a metric that shares an assumption with the bug is blind to it. It
happened anyway. **Build the instrument from what the EYE does, not from what the repair
believes it is doing.**

## 3. Everything that was found, was found by looking

None of these appeared in any number, and two of them were my own "fixes":

* **The white tetris block.** At source 721 left the lash minimum is at patch row 14 and
  **the highlight starts at row 15-17**, so the fixed `+2 px` guard offset landed inside
  the specular and sheared off its corner.
* **The footprint exemption put lashes back.** Its justification — *the footprint is where
  the plate is brighter than the canonical iris, and a lash is darker* — is FALSE once the
  lid shadow is modelled: the shadowed canonical iris is then darker than a lash, so the
  detector marks lash pixels as excess. Measured: 1333 pixels changed above the lash line,
  **70 % of them dark**.
* **The lash-inner-edge walk made it much worse.** Source 736-745 came back with a large
  white blob, because the walk ran down through the merged dark run of §0 and put the
  ceiling below the specular.
* **The black crescent.** Exempting only the bright lobe left the specular's coding
  undershoot standing — a dark ring round the top of the pupil at 736-745, which is
  exactly what rounds 32-34 were rejected for.
* **The lid shadow's halo** — §4.

## 4. The LID SHADOW is rejected on sight and is now DEFAULT OFF

Round 35 §7 asked for it; this round fitted it (`1 + A·exp(−depth/tau)` in depth below the
fitted lid margin, ONE shape per eye for the shot, only a scalar amplitude per frame;
A = **+0.4396**, τ = 3.2 px left, A = **−0.1150**, τ = 5.2 px right, 17 000 samples per
eye). One look at the face-scale panel killed it.

At +0.44 it puts a **light washed band across the top of the iris** where her source and
r35 both carry a dark lid shadow — it **erased the very thing the artist asked to keep**:

> it's both correct to preserve a shadow from the bottom of the eyelid along the arc of
> the eye for natural depth and separation

and that bright band above the pupil is what made the pupil read as displaced and
misshapen — part of his *"the pupil and iris still shifts around"*.

**Why the number and the picture disagreed, which is the reusable part.** The pooled depth
profile behind it is real: over the shot, with the specular and pupil excluded, the fill IS
too dark 3-5 px below the lid margin (left eye, result − source: −0.090 at depth 3, −0.149
at 4, −0.050 at 5). But applying that as a per-eye constant over-brightens every frame that
was already right, and a 20 % lift over a band a few pixels wide reads as a halo.
**A correction like this must be validated at FACE SCALE on frames before it is believed;
a pooled profile cannot see a halo.**

`--lid-shadow` defaults to 0. The fit is still computed and written to `removal.json` so
round 37 can use the measurement without the mistake.

## 5. What is real and worth keeping

* **`lash_ceiling()`** — a per-column lash line read from her own photograph, applied LAST,
  after the fill, after the removal's excursion past the fitted lid and after the footprint
  union, so it outranks all three. It can only ever REMOVE paint, so it cannot introduce a
  wrong-anchor artefact. This is the right *shape* of solution; its input is what fails.
* **Why a guard and not a better lid fit:** `eyelid_fit.py` is audited to 0.43 px and is
  still confidently wrong where it matters — **seven pixels high on the LID CREASE at
  source 727 left, abstaining entirely at 723 left**. Both are frames he flagged.
* **The ceiling is temporally stabilised** by a three-frame median in the eye's own
  coordinates. Read per frame it moved up to **0.8-0.9 iris radii (10 px) between adjacent
  frames** — that is his *"temporal instability in the colors in the region that changes"*.
  A median of three adds no lag on smooth motion (the median of a monotone triple is the
  middle sample) while outvoting a frame whose argmin flipped a row.
* **The `ncc = 0.0` veto is fixed.** `atl.build`'s `register()` returns None when it finds
  no overlap and **leaves `ncc` at its initialiser** — not a measured correlation. Source
  843, 846 and 848 left carried exactly 0.0 between neighbours at 0.985/0.988/0.975 and
  were declined outright, leaving the whole softbox in frame: his *"glimpses of the
  original softbox in the last seconds"*. With the registration FROZEN every frame uses the
  same median offset, so an unregistered frame is placed as well as its neighbours.
  Untreated open eyes 3 → 0.
* **`review_r36_atlas.py`** — the all-frames atlas he asked for by name, source above
  result at eye scale, defects burned onto each tile.

Verification for `r36g`: 150 frames, exact source outside the reported weight, **300 exact
keyed-eye fallbacks**, **304 protected hashes unchanged**. `production_enabled` and
`artist_accepted` are false.

## 6. The tracker — reproduced, diagnosed, NOT fixed

His words: *"there are times where she looks a bit down and our iris didn't follow along at
the exact right time"* and *"jump into position ... as it corrects the mask offset
suddenly"*. Against the independent estimator in `audit_r35_limbus.py`:

| | left | right |
|---|---:|---:|
| median &#124;offset_x&#124; | **0.05 px** | 0.05 px |
| median &#124;offset_y&#124; | **0.42 px** | 0.50 px |
| worst offset_y | **−4.94 px** | −4.53 px |

**The error is almost purely VERTICAL, always negative, and every excursion is on a frame
where the lid hides the top of the iris** (15-24 edge points against a typical 41). The
geometry explains it: with the radius held at the shot constant, an arc that is only the
sides carries almost no information about the centre's HEIGHT, so the circle slides DOWN
and snaps back when the lid opens.

`--trust-se` (default 0.35) shrinks a frame's centre toward its neighbours in proportion to
`centre_se_px`, the fit's own standard error along the worst-determined axis. **It barely
fires** — 6 frames left, 21 right, max move 0.27 px — because **the fit is confidently
wrong on those frames**. Its own uncertainty estimate does not flag them. Same failure mode
as the eyelid fit on the lid crease.

## 7. Reproduce

```powershell
$P = "dependencies/CorridorKey/.venv/Scripts/python.exe"
$S = "music_videos/reporter_test/shots/reporter_a"

& $P -B scripts/limbus_fit.py  --shot $S      # --trust-se 0.35, barely fires
& $P -B scripts/eyelid_fit.py  --shot $S
& $P -B scripts/eye_soft_removal.py --shot $S --name r36g `
     --atlas-frames "$S/composite/r34_gen_wide/inpaint" --atlas-mode replace `
     --fill-iris 0.99 --grain 0 --pupil-target 0.34      # --lid-shadow defaults to 0
& $P -B scripts/verify_r34.py       --shot $S --name r36g
& $P -B scripts/review_r36_atlas.py --shot $S --name r36g   # all 300 eyes
& $P -B scripts/review_r36_lash.py  --shot $S --side left   # the guard, drawn on frames
& $P -B scripts/review_r35_eyes.py  --shot $S --name r36g_eyes `
     --dirs "r36g/source,r35_final/removal,r36g/removal" --labels "SOURCE|R35|R36g"

# the one that actually finds things -- two results in ONE tile, every frame
& $P -B scripts/review_r36_ab.py --shot $S --side left `
     --dirs "r35_final/removal,r36g/removal" --out "$S/composite/r36_ab/ab"
```

Reviews: `final/r36g_eyes.mp4`, `final/r36g_panel.mp4` (SOURCE / R35 / R36g).

## 8. Starter prompt for round 37

> Stay on eyes. Read round36.md §1 first, then §0.
>
> **The artist's last words on round 36 were "Apparently you can't see it."** Round 36
> repeatedly called frames clean that he could see were not, at 18x, after being told to
> look. Treat every visual claim in that handoff as unverified. Start by putting R35 and
> the current result in the SAME tile, one above the other, at high zoom, stepping every
> frame — and form your own view of what is wrong before reading my diagnosis of it.
>
> **Do not build a seventh threshold variant.** §0 tabulates six with what each one cost.
> The measurement that kills the whole family: at source 740 left her lash and her
> lid-shadowed upper iris are ONE continuous dark run at 0.16-0.22, and the specular's
> falloff reaches down to 0.30. Lash 0.13-0.35, shadowed iris 0.15-0.30, specular falloff
> 0.30-0.45 — nothing separates them by level.
>
> **The way out is a geometrically correct lash line.** `eyelid_fit.py` is audited to
> 0.43 px and still sits seven pixels high on the LID CREASE at source 727 left and
> abstains entirely at 723 left. Round 35 §3 believed it had fixed the crease failure; it
> had not. Fix the fit on the frames that matter and the guard needs no photometric escape
> hatch at all — no `--lash-bright`, no `--lash-ring`, no `--lash-halo`.
>
> **The same fix probably lands the tracker** (§6): both failures are "the estimator is
> confident and wrong when the lid covers the top of the eye".
>
> **The lid shadow stays OFF** unless it is validated at face scale on frames (§4). The
> pooled profile that motivated it is real and is in `removal.json`; the per-eye constant
> application is what produced a halo and erased the shadow he asked to keep.
>
> Preserve exact keyed fallback, the 300/304 protected checks, generic input support and
> observed identity. Report p95 and max, never a median alone. The scene catchlight stays
> PARKED. No key retuning, no capture requirements, no impossibility claims.
