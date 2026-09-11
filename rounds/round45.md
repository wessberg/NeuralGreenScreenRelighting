# Round 45 handoff — the capture room's band is OFF her hair and the tour's own light makes a new one. The band now MOVES with the light instead of against it. **Artist verdict not yet recorded.**

**Date:** 2026-09-11. Starts from round 44's accepted baseline `r44_m50_s05`. The named defect
was the artist's, the moment the skin work landed: *"we still have the shine in the very top of
her hair. Very clearly from a light hanging above her head."* Mid-round he added the two things
that shaped the build: *"if we move one of our own lights up there, in principle it should be
able to recreate a similar shine, just with our own light placed in the scene. So we still need
to capture the nuanced hair details and so we can re-light the hair and preserve the specular
response, and I guess self shadows, etc"* — and, on the first removal, *"'De-sheened' is alright,
but the middle of her head has a gap like she's bald, and it still shines along the arc/edge of
her head. It also needs to be temporally consistent."* Both of those were real bugs; both are
fixed and the causes are recorded in §2.

He also asked, fairly: *"Will it even be possible for us, in 2026? Hair is super difficult"*.
The answer this round acts on, and §6 holds the evidence for: **general hair reconstruction is
not on the table, and it is not what this needs.** Her hair here is pulled back and flat — a
sheet, not a volume of flyaway strands — and it is a strongly ORIENTED texture, measured:
structure-tensor coherence **0.70** over the region and **0.79** inside the band. That is the
one thing a fibre highlight actually requires, a tangent direction per pixel, and it is there.

**Candidate: `r45_tex15`** (`--hair-sheen r45_sheen --hair-specular 0.35 --hair-texture 1.5` on
top of round 44's accepted command). Videos: **`final/r45_hair_compare.mp4`** (R44 left, R45
right) and **`final/r45_panel.mp4`** (face scale, three ways: R44, removal only, removal +
new highlight). Sheets in `composite/panels/r45_*.png`.

---

## 0. What was built

| script | what |
|---|---|
| `hair_region.py` | where the hair is and which way its strands run, both read from the plate. Region = solid matte AND off the (dilated) skin mask AND outside the face oval AND inside `radius` FACE RADII of the oval's centroid; `structure_tangent` is a GPU structure tensor on the log luminance returning a unit tangent and a 0..1 coherence; `tangent_world` lifts that into Blender world axes and projects it onto the neural normal |
| `hair_sheen.py` | the removal. Two passes over the clip: the band's ENVELOPE per frame, then a symmetric head-stabilised temporal median of it, then `S = max(env - b, 0) * fade(env/b) * tex` taken along each pixel's own chroma. Writes `sheen/` (16-bit linear), `field/` (tangent angle, strand texture, coherence) and `weight/`. Also fits and reports a DIAGNOSTIC room-light direction |
| `composite_ratio.py --hair-sheen` | subtracts `sheen/` from the plate before the ratio and adds an anisotropic Kajiya-Kay highlight from each light pass along the measured tangent. `--hair-specular --hair-roughness --hair-f0 --hair-azimuthal --hair-texture --hair-lobe-blur --hair-coherence --hair-clamp --hair-sheen-gain` |
| `meso_specular.kajiya_kay_specular` | `F(v.h) * (p+2)/(8pi) * sin(T,H)^p * max(n.h,0)^q * pi E_k`, lobe widened by the source's angular radius exactly as the GGX term is |
| `probe_r45_hair.py` | the measurement that came before any of it (level, strands, geometry, footprint stability) |
| `audit_r45_hair.py` | the acceptance test, and why round 44's own statistic cannot be read straight here |
| `review_r45_hair.py` | crown-scale stills of several runs, optionally beside the measured fields |

Defaults elsewhere are unchanged: without `--hair-sheen` the composite is round 44's exactly.

## 1. The result, measured

`audit_r45_hair.py`, on the **lit half of the tour** — the 75 frames where the question is
answerable at all (§3 says why the other 75 are not):

| run | corr(round-44 sheen, log E) | corr(specular fraction, log E) | corr(bright-x, sin azimuth) |
|---|---|---|---|
| `r44_m50_s05` (accepted baseline) | −0.034 | +0.075 | **−0.443** |
| `r45_rmonly` (band removed, nothing added) | +0.005 | −0.036 | −0.546 |
| **`r45_tex15` (the candidate)** | **+0.392** | **+0.587** | **+0.518** |

**Round 44's stated acceptance criterion — that corr(sheen, log E) go POSITIVE on the hair — is
met.** And the third column is the one to trust most, because it is the least circular test
available: the source's azimuth comes from the render rig's own manifest, the brightest
decile's position comes from the output picture, and the two share no code and no assumption.
**The bright part of her hair used to move AGAINST the light (−0.44). It now moves with it
(+0.52).** That is the artist's *"natural, authentic response"* in one number.

Removal alone does not do it (row 2): taking the room's band off makes the hair darker and the
statistics slightly worse. **The two halves are one change** and neither is a result on its own.

## 2. The two bugs the artist saw in one still, and what actually caused them

He looked at the first `de-sheened` frame and named both immediately. Neither was a tuning
problem; both were structural, and the fixes are the reason the second build works.

**a. "It still shines along the arc/edge of her head."** Two independent causes, both about
edges. The region required `alpha > 0.99`, so the fractional matte pixels along the silhouette
— where the brightest sheen actually sits — were outside it; and the weight was feathered on
ALL of its edges including that silhouette, fading the removal out exactly where it was most
needed. The region now runs to `alpha > 0.5` and is **weighted by alpha**, and the feather is
applied only to the ARBITRARY cuts (hairline, face oval, radius). *Then* the deeper one: the
baseline was a LOCAL low percentile over 0.30 face radii ≈ 54 px, inside a band ~200 px across,
**so the band was setting its own baseline and reporting no excess**. The baseline is now the
REGION-WIDE percentile, which is the statistic round 44 §5 measured its 23× against.

**b. "The middle of her head has a gap like she's bald."** Her parting is scalp, so the SKIN
MASK claims it, and subtracting the skin mask cut a hard strip down the middle of the crown
that no sheen stage could touch — leaving it at plate brightness over a crown that had just
been darkened. Scalp showing THROUGH hair is enclosed by hair on both sides, so a morphological
closing at `close_scalp` face radii bridges it while leaving the hairline (hair on one side
only) exactly where it is. Two smaller things went with it: the first chroma gate (full removal
≤ 12°, none ≥ 25°) punched the same hole photometrically and is now much wider (20°/45°), and
subtracting along the ILLUMINANT's chroma left a **red streak** down the parting, because scalp
is skin at 30° from the illuminant and a neutral subtraction leaves the red behind. The
subtraction is now taken along **each pixel's own chroma**, which cannot shift a hue by
construction — and round 44 §5 already measured that on hair the two are interchangeable (band
and dark hair both 8–9° from the illuminant).

**c. A mask that does not know the face is not a hair mask.** The first version was "solid matte
and not skin", and it handed back her irises, lashes, nostrils and lower lip as crown sheen —
the skin mask excludes all of them. The face oval, padded by 0.07 face radii to clear the brows,
is a separate and necessary cut. And the skin mask stops INSIDE the hairline, so without
dilating it by 0.025 face radii a rim of lit forehead came back as "hair": **37 % of the >4×
band's pixels and a third of its energy**, all of it at skin chroma.

## 3. Round 44's metric cannot be read straight once the band is gone, and the number that says so

`audit_r44_sheen.py` reports `sheen = p95/p50` of the region and correlates it with log E. The
ratio is deliberately invariant to a global multiply — that is the whole point, since `plate × R`
rescales a photographed highlight along with the hair under it. It works while the denominator
is a real number. **Subtracting the band makes it one.** At the dark half of the tour the
composite already puts the whole hair region on its own floor (median luminance 0.00003 against
0.011 at the bright stations), so p95/p50 becomes the ratio of two floor values: the audit
reports a "sheen" of **15 168** at the dark stations against 11.9 at the bright ones, and that
drags the whole-tour correlation to −0.50 no matter what the highlight does.

So the whole-tour figure is reported (and is negative) *and* the lit half is reported separately,
which is where §1's table comes from. `audit_r45_hair.py` prints both and says which is which.
**Do not quote the whole-tour number as a failure or the lit-half number as the whole story.**
A `specular_fraction` statistic — the share of the region's light standing above its own p20 —
was added because it is bounded in [0,1] and cannot explode; it degenerates at the dark stations
too (it saturates at 1.000), so it is only useful on the lit half either.

## 4. The replacement, and the three things that were wrong with the first one

**a. Kajiya-Kay alone glows everywhere, and this is structural, not a tuning error.** `sin(T, H)`
is 1 for EVERY tangent lying in the surface whenever the half-vector is near the surface normal,
so a front-lit front-viewed head is uniformly at the lobe's peak. Measured: the added highlight
hit its clamp on the MEDIAN hair pixel and the crown read as a lit wig. A real fibre is a
cylinder and only the part whose own normal faces the half-vector reflects, so
`max(n·h, 0) ** --hair-azimuthal` (default 2) was added. **That is what makes it a band.** With
it off the crawl is also worse (§5), which is a second reason to keep it.

**b. Gating on coherence punches holes.** The structure tensor returns one direction per pixel
with a confidence, and zeroing the lobe below a threshold left **dark blobs in the crown**. A
pixel covers many fibres, so the lobe is now averaged over a small neighbourhood WEIGHTED BY THE
COHERENCE (`--hair-lobe-blur`, 0.008 face radii): confident neighbours fill in for pixels where
the direction is unknown. The blobs are gone.

**c. A smooth lobe on dark hair reads as HAZE, not as hair catching light.** The tangent field is
smooth, so the lobe is, and the result was a grey milky wash — the hair equivalent of round 44's
"smooth band is the Barbie signature". The plate's own strand modulation breaks it up
(`--hair-texture`). Unlike `--specular-texture` on skin, **this one is normalised by its own
low-pass**, so the exponent changes the CONTRAST of the streaks and not their level — which is
what lets the gain stay calibrated (§4d) while the streakiness is chosen separately.

**d. The gain is MEASURED, not chosen.** The band being replaced implies a hair specular
reflectance: a highlight is `f_r × π × E`, and `E_room` on those pixels is already measured
(`estimate_source_light.py`), so `S / (π E)` is `f_r` at the geometry the band was made at —
which is at or near the lobe's peak, since that is what a band IS. `hair_sheen.py` logs it as
**`fr_peak_measured` = 0.076** (p50 over the clip). The model's own peak at F0 0.0465 (cuticle,
IOR 1.55) and p = 76 is `f0 (p+2)/(8π)` = **0.144**. `0.076 / 0.144 = 0.53`, so
**`--hair-specular` defaults to 0.55**. The candidate runs at 0.35, below that, on the stills —
his call on the video.

## 5. Temporal: measured, decomposed, and one term is the whole of it

Frame-to-frame step in the hair region, head-registered, p95 per frame, median over the clip:

| run | median of frame p95 (stops) | vs baseline |
|---|---|---|
| `r44_m50_s05` | 1.622 | — |
| `r45_rmonly` (removal only) | 1.706 | +5 % |
| `--hair-texture 0` | **1.533** | **−6 %** |
| `--hair-texture 1.5` (**the candidate**) | 1.822 | +12 % |
| `--hair-texture 3` | 2.073 | +28 % |
| `--hair-azimuthal 0` | 2.214 | +36 % |

**The removal adds essentially nothing** (+5 %), which is the point of solving the envelope once
per window instead of per frame: `hair_sheen`'s own report puts the de-sheened plate's step at
**0.914 stops p95 against the untouched plate's 0.865 on the same pixels** — it inherits the
photograph's temporal behaviour rather than adding to it.

**All of the added crawl is the strand-texture term**, and that was worth decomposing rather
than guessing: the tangent field was ALSO filtered in time this round (a symmetric
head-stabilised window on the coherence-weighted DOUBLE-ANGLE vector, rotated by twice the
warp's own rotation — a direction field is not an image) and it moved the number by 0.011 stops,
i.e. **the tangent was never the problem**. Neither are the neural normals: turning the
azimuthal term off, which removes the normals from the lobe entirely, makes it *worse*. If the
artist reports crawl, the lever is `--hair-texture`, and the sweep above is the price list.

## 6. What this cannot do, and why — so no round re-derives it

* **Individual flyaway strands at the silhouette.** Sub-pixel, living in the matte's alpha ramp
  (round 43: 98.5 % of alpha is exactly 0 or 1, and the ramp is 1 px).
* **Hair-on-hair self-shadowing.** The artist asked for it explicitly. The proxy shell has no
  strand geometry, so there is nothing to cast; the plate's own strand texture is a stand-in
  that carries the *appearance* of inter-strand occlusion frozen at the capture room's light
  direction, and it is a HEURISTIC, exactly like `--specular-texture` on skin. Real self-shadow
  needs strand geometry in the renderer, which is the thing monocular hair reconstruction cannot
  give in 2026.
* **The hair changing SHAPE with the light.** Not attempted; it cannot be right.
* **Marschner's TT and TRT lobes**, and the cuticle tilt that offsets them. This is the primary
  (R) highlight only. On near-black hair very little light returns from inside the fibre, so R
  carries nearly all of the visible sheen — which is why the approximation is defensible here
  and would not be on blonde hair.
* **The physical room-light fit is a DIAGNOSTIC, not the removal, and this was measured.** A
  single Kajiya-Kay lobe with five parameters for the whole clip correlates with the band's
  envelope at only **0.45** over 60 000 crown samples from five stations — about a fifth of its
  variance. It does establish something that was otherwise unknown and is worth keeping (the
  room's source sits **above and in front of her**), and it is still run and still logged as
  `room_light_fit`. **Do not promote it to the subtraction.**

## 7. Reproduce

```powershell
$P = "dependencies/CorridorKey/.venv/Scripts/python.exe"
$S = "music_videos/reporter_test/shots/reporter_a"
& $P -B scripts/probe_r45_hair.py --shot $S                      # the measurement, before anything
& $P -B scripts/hair_sheen.py --shot $S --name r45_sheen --length 150       # ~150 s, GPU
$C = "--shot $S --eye-removal r42_wet_removal --corneal-dir r42_eye_pass --corneal-blur 0.35 --corneal-matte lids --light-passes bulb=r44_light_bulb,fill=r44_light_fill --meso-sigma 0 --meso-strength 0.5 --specular-model physical --roughness-skin 0.55 --f0-skin 0.028 --specular 0.5 --specular-texture 2 --specular-clamp 8"
& $P -B scripts/composite_ratio.py $C.Split() --hair-sheen r45_sheen --hair-specular 0 --name r45_rmonly
& $P -B scripts/composite_ratio.py $C.Split() --hair-sheen r45_sheen --hair-specular 0.35 --hair-texture 1.5 --name r45_tex15   # THE CANDIDATE
& $P -B scripts/audit_r45_hair.py --shot $S --dirs r44_m50_s05,r45_rmonly,r45_tex15
& $P -B scripts/review_r45_hair.py --shot $S --dirs r44_m50_s05,r45_rmonly,r45_tex15 --labels "R44|removal only|R45" --frames 73,97 --side crown --out r45_crown.png
& $P -B scripts/review_video.py --shot $S --name r45_tex15 --compare r44_m50_s05 --label "R45: hair" --compare-label "R44: accepted" --out r45_hair_compare.mp4
& $P -B scripts/review_r37_panel.py --shot $S --name r45_panel --dirs "r44_m50_s05/.,r45_rmonly/.,r45_tex15/." --labels "R44 accepted|R45 removal only|R45 removed + new highlight"
```

## 8. Starter prompt for round 46

> **Round 45's verdict is the first thing to get.** The hair work is built and measured but
> **the artist has not judged the video** — `final/r45_hair_compare.mp4` and
> `final/r45_panel.mp4`. Do not treat §1's numbers as acceptance. If he rejects it, read §4
> before touching a knob: each of the three failures there was structural, and the fix was never
> the gain. The levers, in the order they matter: `--hair-texture` (streakiness vs crawl, priced
> in §5), `--hair-specular` (0.55 is the MEASURED value, 0.35 is the candidate),
> `--hair-azimuthal` (how much of a band it is).
>
> **The LIP GLOSS is the same family and is still open**, and round 44 §5 has already measured
> it: sheen 2.1 at the dark stations against 1.4 at the bright ones, corr −0.39, and the
> face-delight separation takes only 9–15 % off. It needs the band itself — no low-pass, the
> DiffusionRenderer vermilion as the diffuse reference, and a check that the a-term is not
> eating lip colour. `hair_sheen.py`'s shape applies almost directly (an envelope over a
> region-wide baseline, filtered in time, subtracted along each pixel's own chroma) but the
> lip's specular is a SURFACE lobe, not a fibre one, so the replacement is the existing
> `--specular-model physical` with a lip-specific roughness rather than Kajiya-Kay.
>
> **Do not re-derive, and do not re-open:** the crown band is 23× the dark hair at the SAME
> chroma so no dichromatic split can touch it (round 44 §5); `face_delight --regions hair` does
> not move the audit; UnReflectAnything was ruled out on skin in round 43 and is not worth
> re-evaluating on hair; render settings are not a lever for this void scene (round 44 §4c);
> and the single-lobe physical fit explains only a fifth of the band (§6).
>
> **Still open and still needing HIS decision, unchanged from round 44:** the fill panel
> reflecting in a shadowed eye is the RIG parking the fill BEHIND her face plane at the side
> stations (round44 §6) — fixing it re-renders the tour, both light passes AND the accepted eye
> pass, so ask before doing it. And porting `--only-light` to Blender's native LIGHT GROUPS
> halves the render cost; gate the port on reproducing `--only-light`, not on bit exactness.
>
> **Two lessons from round 45 worth keeping.** First: **a metric can be invalidated by the fix
> it is judging.** Round 44's p95/p50 sheen ratio was the right statistic for a photographed
> band and became a ratio of floor values the moment the band was subtracted — 15 000 at the
> dark stations. When a number moves the wrong way after a change that visibly worked, check
> whether the change broke the number's denominator before believing it. Second: **when an added
> term crawls, decompose before filtering.** The tangent field was filtered in time at some cost
> and moved the crawl by 0.011 stops; the whole of it was the strand-texture exponent, which one
> ablation found.
>
> Keep: GENERIC tool, GPU first, plate pixels outside the pass exact, p95 AND max for anything
> temporal, a labelled crown-scale sheet in front of him before a full run, and a video before a
> verdict.
