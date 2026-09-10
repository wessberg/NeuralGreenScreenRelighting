# Round 37 handoff — accepted on the frames shown; next, the whole 20 s clip

**Date:** 2026-09-10. Starts from round 36 (rejected). Production defaults unchanged —
every new behaviour is behind a flag, and the full command is in §7.
**Artist verdict on `r37w`:** *"Based on just the presented frames, it seems acceptable enough
now."* Before that, on R37k: *"It's almost there!"* and on R37h: *"It's a lot better, closing
on an acceptable solution."*

**"The presented frames" is the 150-frame selection, source 700–849, out of a 1200-frame,
20.02 s clip.** Nothing here has been run or looked at outside that range. Round 38's job is
to find out where it breaks on the rest (§8).

---

## 0. What the artist asked, and what it turned out to be

His opening complaint about R36g: *"we still see the outline of the softbox reflection
sometimes, and other times there are small color variations within the spot we replaced it
with, and it's not always cleanly looking like a round-ish pupil inside an iris."* Looking at
every frame (tight stacked crops, §6), all three came down to two causes:

1. **The lash guard cut holes in the softbox.** `lash_ceiling()` walks each column down from
   the lid skin, skipping the blocked highlight and pupil. In every column the softbox crosses,
   the walk jumps over it and lands in the LOWER iris, and a median along x repeats that error.
   So the fill was a U with the softbox in the notch, patched pixel by pixel by the three
   photometric exemptions (`--lash-bright/ring/halo`). Whatever they missed was the softbox's
   falloff at 0.30–0.45 display, which is the grey OUTLINE. The patchwork of exempted and
   unexempted pixels was the "colour variations within the spot". Drawn on frames:
   `composite/r37_lash_look/` (green = ceiling).
2. **The canonical iris was contaminated in its top half.** Vote counts per canonical cell:
   **~128 frames** in the band from the eye centre to 0.6 radii below it, **6–19** above the
   centre, **0** at the top, because the lid covers the top on most frames. A per-cell median
   of six frames is whatever those six frames held. Looked at (`canonical_<side>.png`,
   `canonical_<side>_votes.npz`), the left canonical had a **dark band across the top of the
   pupil**, a mauve lid-skin patch and diagonal lash streaks. The right one had straight mask
   edges and a reddish patch. Every repaired frame paints from this one image, so this was in
   every frame: the non-round pupil and more of the colour variation.

## 1. The construction in R37w

Each item is a flag. The docstrings carry the measurements.

| piece | flag | what it does |
|---|---|---|
| softbox region | `--specular-reach 2` | `corneal_specular()` finds the one compact bright blob with a DARK surround, dark above too, and `specular_region()` replaces it plus a 2 px disc WHOLE, after every guard. Distance, not level: the codec and lens blur a highlight's edge over a fixed couple of pixels. |
| smooth lash line | `--lid-line parabola --lid-line-window 2` | `lid_line()` fits the shot's lid curvature through the trustworthy walk samples only (not the softbox's columns, not below the eye centre), then takes a median of 5 frames in eye coordinates. |
| no exemptions | `--no-lash-exempt` | the softbox is protected by geometry, so the photometric exemptions only added patchwork |
| clean canonical | `--canonical-votes 60` | `symmetric_canonical()`: a cell seen by fewer than 60 frames is rebuilt, in proportion, from its mirror across the horizontal diameter (a ring maps onto itself, so the pupil comes out round), falling back to her measured radial profile |
| pupil colour | `--pupil-colour` | the pupil is darkened per channel to the colour of her own darkest pupil pixels; one scalar had left it a saturated red-black |
| edge shading | `--top-shading 2.5 --shading-sides all` | `top_shading()`: the fill takes her own shading from the source pixels just outside every interior edge, fading over 2.5 px. It never reads the softbox (`exclude`) or anything brighter than 0.45 display. |
| region above the lid line | `--region-rows` | `region_rows()`: row by row from her pixels left and right at the same height, ignoring pixels still carrying the reflection's falloff |
| temporal shading | `--shading-temporal 2` | `filter_shading()`: two-pass render. The shading field gets a 5-frame median at the same place on her eye. Only frames that PAINTED a spot vote on it, and the window is always SYMMETRIC (lag-free). |
| level | `--gain-ring lower --gain-temporal --level-window 2` | the per-frame gain is read from the UNSHADOWED lower iris, median of 5, and bridged instead of falling back to round 33's ramp |
| lids | `--lid-temporal` | `load_lids()`: keeps a lower lid whose upper failed. Bridging is only BETWEEN fitted frames. Median of 3 of the height, measured against the eye corners. |
| limbus slide | `--limbus-gate 1.0` | `gate_limbus()`: iterative, worst first. A frame whose limbus breaks from its neighbours while the eye corners don't takes the neighbours' offset. 4 eye-frames trip it (704 L, 843 L, 716–717 R). |
| no upper bound | (with `--lid-temporal`) | no fitted upper lid and no lash line means no whole-iris fill and no excess removal, only the softbox's own region |

## 2. Measurements (`audit_r37_eyes.py`, all 273 eyes; `audit_r37_flicker.py`, steady stretch)

| | R35 | R36g | R37k | **R37w** |
|---|---:|---:|---:|---:|
| bright fragments left (eyes / px) | 115 / 252 | 182 / 647 | 145 / 303 | **94 / 130** |
| softbox region standout, median / eyes > 0.05 | −0.002 / 9 | +0.006 / 14 | +0.057 / 147 | **+0.006 / 0** |
| upper-iris level vs source, median / p95 abs | −0.026 / 0.094 | −0.067 / 0.098 | −0.020 / 0.043 | −0.020 / 0.045 |
| lower-iris level, p95 / max abs | 0.049 / 0.106 | 0.050 / 0.088 | 0.037 / 0.067 | **0.020 / 0.031** |
| lash px changed (lightened) | 1342 (553) | 208 (22) | 44 (2) | 63 (17) |
| departure from her own change, p95 / max | 0.050 / 0.159 | 0.045 / 0.162 | 0.055 / 0.166 | **0.037 / 0.095** |

Source 746–778 in eye coordinates, left / right. R37h is what the artist saw when he called
the flicker:

| | fill-wide L* step p95 | fill-wide chroma step p95 | local L* flicker | local chroma flicker |
|---|---|---|---|---|
| source (control) | 0.47 / 0.50 | 1.22 / 1.37 | 0.77 / 0.99 | 1.66 / 2.04 |
| R36g | 0.32 / 0.48 | 1.14 / 1.02 | 0.29 / 0.27 | 0.59 / 0.60 |
| R37h | 0.89 / 0.56 | 0.85 / 0.72 | 0.60 / 0.52 | 0.66 / 0.66 |
| **R37w** | 0.31 / 0.17 | 0.52 / 0.55 | 0.36 / 0.31 | 0.58 / 0.57 |

Local L* flicker in R37w is still ~20 % above R36g's, in a thin band where the fill's edges
blend into her codec-noisy source. The interior is as still as R36g's.

Verification for `r37w`: 150 frames, exact source outside the reported weight, **300 exact
keyed-eye fallbacks, 304 protected hashes unchanged**. `production_enabled` and
`artist_accepted` are false.

## 3. Everything the artist caught, and what it was

* **R37h: "quite a bit of flicker … small color variation here and there … not as easily
  detectable in your atlas tables".** The new edge shading read her noisy edge pixels afresh
  every frame. Local L* flicker doubled and the fill-wide level pulsed (p95 0.89 L*). **No
  static tool could see it**; `audit_r37_flicker.py` (stabilised, CIELAB, global vs local)
  and `review_r37_temporal.py` did at once. Fix: `filter_shading()`.
* **R37k: a tan block above the pupil at 717 right** (his screenshot). There were three
  causes. The softbox there sits in the lash line, so its region was painted with canonical
  iris above the lid line. The temporal median then let neighbours that hadn't painted that
  spot vote "no shading". And the edge shading read the SOFTBOX as her shadow, brightening the
  fill under it. The last one also put a faint orange arc under the lash on most open frames
  of the right eye (756–779). On top of that, the limbus fit **slid 3.7 px down on that one
  frame** (−3.0 / +0.7 / −3.0 px from the eye corners across 716–718). That put the pupil low
  and let the fill run onto the lower lid.

Found by looking before he did, and fixed:
* **704/705 left:** iris painted onto a closing lid. A neighbour's lash line or lid had been
  copied from ONE side, and at a blink edge that neighbour is always the more open frame.
* **712/713 left:** a brown block on the lid SKIN. The excess-based removal, with no lash line
  left to cut it, took bright skin for "brighter than the iris". I introduced this myself
  mid-round and caught it on the sheets.
* **R37b: a dark cap on the pupil** from taking shading off the lash above.
* **R37m: a grey smudge at 718** from an RGB heal of the region, which carried the softbox's
  faint falloff colour in.
* **835–838: a dark cup in the lower iris after a blink.** The level fell back to round 33's
  ramp when the iris ring was under 40 px.
* **843–847 left: two pupil colours** where the lid line crosses the pupil.

## 4. Tried this round and rejected, so it is not tried again

| try | what happened |
|---|---|
| shading taken from the pixel directly above the softbox region (R37b) | lash-black cap on the pupil, lash strands as vertical streaks |
| RGB "healing" fill of the region above the lid line (R37m) | fixed 717, painted a grey smudge at 718 |
| membrane with a no-flux top (R37c–k) | the region's top came out lighter than the lash beside it; the 717 tan block |
| temporal median, one-sided at run ends (R37i) | a lighter band under the lash the frame before a blink (826): lag |
| lid-registered temporal filter (R37l) | no gain on either problem |
| soft (anti-aliased) lid edges (R37j) | no effect on the bottom-edge dashes, and a worse outlier |
| two-sided-only lash-line bridging | bridged nothing; 23 squint eye-frames lost the fill |
| limbus gate on the offset from the corners alone | imported mediapipe CORNER jitter into a correct limbus (764–770 left) |
| limbus gate against the neighbours' mean | one slid frame dragged both neighbours past the gate |

## 5. Metrics that were wrong this round, and how

* **Round 36's lash count** measured paint "above the lash line" against the same ragged
  ceiling that drops to the lower iris under the softbox, so it counted the filled pupil as
  lash (9058 px on R37c). The replacement reads the walk only in the columns it can be trusted.
* **The fragment test** counted lid-skin pores on squint frames (bright on bright inside
  rho < 0.9). A leftover reflection is bright on a DARK surround, and the test now requires
  that.
* **Per-pair flicker on the limbus-track grid** reported R37r's worst "flicker" exactly on
  the frames where the gate had FIXED the slide. R37q's fill followed the slid track, so it
  looked still in track coordinates while jumping 3.7 px in the picture. Checked in
  corner-stabilised crops, it was the reverse. **Any measurement that rides the tracker
  inherits the tracker's error.**
* **The departure-from-source measure** first flagged a ring round the softbox. That was the
  SOURCE's own softbox ringing moving, which R37 replaces and R36g didn't.

## 6. Still open

All of these are single frames at the edge of a blink, except the last three:
* **713 left / right:** a dim remnant of the softbox under the closing lid (peak 0.28 on a
  0.09–0.17 surround). At that level it joins her lid skin into one component, so it can't be
  isolated.
* **826 right** (the frame before a blink): the deep lid shadow right of the pupil comes out
  0.04–0.08 lighter than her near-black source. That is the audit's 17 lightened lash pixels.
* **704 left, 714/715 right:** a black canonical pupil under a nearly shut lid, darker than
  her lid-shadowed one. This is the most negative region standout (−0.21 at 704 left).
* **Bottom-edge dashes, 837–842 right:** the fill's lower edge moves a row against her lower
  lid margin. R36g has them too.
* **The limbus tracker's vertical slide is patched, not fixed.** The gate catches one-frame
  slides; a slow slide over several frames would pass it.
* **Only 150 frames exist** for every per-frame asset. `eye_soft_removal.py` holds every
  frame of the shot in RAM as full-frame float32 (sources + pooled), about 25 MB each. That
  is fine at 150 and **about 60 GB at 1200**, so it must hold eye windows only.

## 7. Reproduce

```powershell
$P = "dependencies/CorridorKey/.venv/Scripts/python.exe"
$S = "music_videos/reporter_test/shots/reporter_a"

& $P -B scripts/eye_soft_removal.py --shot $S --name r37w `
     --atlas-frames "$S/composite/r34_gen_wide/inpaint" --atlas-mode replace `
     --fill-iris 0.99 --grain 0 --pupil-target 0.34 --pupil-colour --canonical-votes 60 `
     --limbus-gate 1.0 --lid-temporal --lid-line parabola --lid-line-window 2 `
     --specular-reach 2.0 --no-lash-exempt --region-rows `
     --top-shading 2.5 --shading-sides all --shading-temporal 2 `
     --gain-ring lower --gain-temporal --level-window 2
& $P -B scripts/verify_r34.py --shot $S --name r37w

# the numbers: all eyes, and flicker on a steady stretch
& $P -B scripts/audit_r37_eyes.py    --shot $S --names "r36g,r37k,r37w" --out r37_eyes_audit_final.json
& $P -B scripts/audit_r37_flicker.py --shot $S --names "r36g,r37w" --side left --frames 46-78

# the looking: stacked tight crops of every frame, stabilised frame differences, videos
& $P -B scripts/review_r37_sheets.py   --shot $S --side left --dirs "r37w/source,r37k/removal,r37w/removal" --labels "SRC|R37k|R37w" --out composite/r37_sheets/left
& $P -B scripts/review_r37_temporal.py --shot $S --side right --frames 14-20 --dirs "r37w/source,r37k/removal,r37w/removal" --labels "SRC|R37k|R37w" --mode lightness --gain 12 --out "$S/composite/r37_temporal/r714.png"
& $P -B scripts/review_r35_eyes.py     --shot $S --name r37w_eyes --dirs "r37w/source,r37k/removal,r37w/removal" --labels "SOURCE|R37k|R37w"
& $P -B scripts/review_r37_panel.py    --shot $S --name r37w_panel --dirs "r37w/source,r37k/removal,r37w/removal" --labels "SOURCE|R37k|R37w" --width 264 --height 170 --drop 30 --zoom 4
```

Reviews (display-sRGB CRF 12, review only): `final/r37w_eyes.mp4`, `final/r37w_panel.mp4`
(SOURCE / R37k / R37w). Earlier ones shown to him: `final/r37h_eyes.mp4`,
`final/r37k_eyes.mp4`, `final/r37q_eyes.mp4`. The canonical eye itself:
`composite/r37w/canonical_left.png` and `canonical_right.png`.

## 8. Starter prompt for round 38

> Stay on eyes. Read round37.md §0, §5 and §6 first. R37w was accepted *"based on just the
> presented frames"* — and those are source 700–849, 150 frames of a 1200-frame, 20.02 s,
> 59.94 fps clip. **The artist's direction for this round: generate the eyes for the ENTIRE
> 20 s clip so we can see where else it breaks.**
>
> Build a full-clip shot rather than widening `reporter_a`: the protected hashes, the 300
> keyed fallbacks and every round's comparison are anchored on reporter_a's 150 frames. Copy
> its `shot.json` to a new shot (e.g. `reporter_full`) with `selection` 0 / 1200. Then
> extract, key with reporter_a's key calibration (same plate — do not recalibrate), face
> track, `limbus_fit.py`, `eyelid_fit.py`. Round 10's `track_r10_source.py` already decoded
> and tracked all 1200 frames sequentially and may shortcut the tracking.
>
> Two things will fail first, and both are known:
> 1. **Memory.** `eye_soft_removal.py` holds every frame as full-frame float32 twice,
>    ~25 MB each: ~60 GB at 1200 frames. Keep only the eye windows (they are ~40×40 px).
> 2. **Atlas material.** `--atlas-frames composite/r34_gen_wide/inpaint` covers only the 150
>    frames. Either regenerate it for the clip (`eye_video_inpaint.py`, ProPainter) or build
>    the canonical from her own clean pixels (drop `--atlas-frames`). `symmetric_canonical()`
>    removes the top-half contamination either way. Try the second on reporter_a first and
>    put it beside R37w before committing to it.
>
> Then LOOK at every frame — `review_r37_sheets.py` stacks source and result in one column
> per frame at a zoom where a pixel is visible — and run `audit_r37_eyes.py` (all eyes,
> including `region_standout`) and `audit_r37_flicker.py` on steady stretches. Expect new
> failure modes: more blinks (§6 lists the blink-edge ones already known), head turns,
> gaze changes, hands near the face. Put frames and videos in front of him early; he judges
> in motion, and the flicker he found in R37h was invisible on every static sheet.
>
> Keep what round 37 established: the softbox region is replaced whole by geometry, never
> cut by a level; shading comes from HER pixels at the fill's edges, filtered in time with
> symmetric windows and painted-only votes, and never read off the reflection or lid skin;
> a neighbour's lid is never copied from one side at a blink; the limbus gate needs the
> limbus to break from its neighbours AND the corners not to. Preserve exact keyed fallback,
> generic input support and observed identity. Report p95 and max, never a median alone.
> The scene catchlight stays PARKED. No key retuning, no capture requirements, no
> impossibility claims.
