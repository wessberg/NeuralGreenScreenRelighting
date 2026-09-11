# Round 45 handoff — ENTIRELY REJECTED. **"Your work this round is entirely rejected."** The band IS the hair's form; every number went the right way and she looked partly bald

**Date:** 2026-09-11. Starts from round 44's accepted baseline `r44_m50_s05`, which **remains
the baseline**: nothing here ships. `--hair-sheen` is opt-in and a composite run without it is
**byte-identical to `r44_m50_s05`**, verified on four stations after the verdict.

**THE VERDICT IS THE MOST IMPORTANT THING IN THIS FILE.** Round 44's starter set the acceptance
criterion as a number — *"the correlation must go POSITIVE"* — and round 45 drove it positive:
§1 has round 44's own statistic going −0.03 to +0.39, and the least circular test available,
the band's position against the rendered source's azimuth, going −0.44 to +0.52. **The artist
rejected it on sight and preferred the untouched photograph to both the removal and the
replacement.** So:

* **The number was satisfied and the bar was not.** Do not re-attempt this because the audit
  says it works — the audit said it worked. This is the standing rule in its sharpest form yet:
  his eye is the acceptance test, and a metric that agrees with a rejected picture was measuring
  the wrong thing. It is not evidence that he is wrong.
* **A photographed band in the WRONG PLACE beats a synthesised band in the right one**, on this
  material at this quality. That finding is worth more than the code.
* **And it is the REMOVAL, not the replacement, that does the damage.** Shown the three columns
  side by side (`composite/panels/r45_three_ways.png`: the accepted R44, the band taken off with
  nothing added, and that plus the new highlight), his verdict was *"everything in
  r45_three_ways looks completely ridiculous, like she's partly bald. Rejected entirely"* -- and,
  asked directly about the first column, *"No, the accepted looks fine."* **So R44 is fine and
  BOTH round-45 columns read as partly bald.** Round 44 SS5 predicted exactly this -- *"the crown
  sheen carries essentially all of the hair's visible form; delete it and the hair is a black
  mass, which is a worse artefact than the one being named"* -- and round 45 read that as an
  argument for building a replacement rather than as an argument against removing at all.
* **There is therefore no safe middle.** Attenuating the band partway (`--hair-sheen-gain 0.3`)
  is the same operation at a third of the strength: it takes away form, and form is what the
  band carries. Do not offer it as a compromise, which an earlier draft of the round-46 starter
  did before his second verdict.
* **The starter's framing was itself the trap.** "Remove AND replace" was written as the
  construction before anyone had seen a replacement. Round 44 §5 said *"delete it and she has a
  black mass, which is a worse artefact"* — true — and then concluded that the answer was to
  synthesise a new band, which does not follow from it.

He asked for this work (*"if we move one of our own lights up there, in principle it should be
able to recreate a similar shine"*) and then rejected the result. Both are true and neither
cancels the other: the idea is sound, and the execution on monocular hair is below his bar.
**Ask before spending another round on hair.**

**Read §6 first if you are tempted to try again** — it is the list of what this class of
approach cannot do, and it was written BEFORE the verdict, not as an excuse after it.

Runs, all preserved: `r45_rmonly` (band removed, nothing added), `r45_tex15` (the candidate),
`r45_hair35` / `r45_hair55` (other gains), `r45_dbg_*` (the ablations). Videos
`final/r45_hair_compare.mp4` and `final/r45_panel.mp4`; sheets in `composite/panels/r45_*.png`.

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

> **Round 45 was REJECTED, and the hair is not to be re-attempted without asking him first.**
> His words: *"what you did to the hair really doesn't look good. The source is way better."*
> Read the top of this file before anything else. The short version: the round drove round 44's
> stated acceptance number from −0.03 to +0.39 and the picture got worse, so **a passing metric
> is not a result here** — and the untouched photograph beat both the removal alone and the
> removal plus a synthesised highlight.
>
> **Do not start a hair round on your own judgement, and do NOT offer "attenuate it a bit" as a
> compromise.** He was shown the removal WITHOUT any replacement, on its own, and rejected it in
> the same breath as the replacement: *"everything in r45_three_ways looks completely ridiculous,
> like she's partly bald"*, and then, on the whole round, *"your work this round is entirely
> rejected"* — while confirming that the accepted R44 column *"looks fine"*. Attenuation is that
> same subtraction at a third of the strength. **The band is the hair's visible form, not a layer
> on top of it**, and any method that takes it off takes the form with it.
>
> So the position to hold, unless he says otherwise, is: **leave the hair alone.** Mark
> constraint row 5 as unmet on hair, with this round as the evidence — that is what *"the source
> is way better"* literally says, and he has now said it three times.
>
> If he ever does want it revisited, the only direction that is not a subtraction is to leave
> every photographed pixel where it is and change only its LEVEL as a function of the new light:
> a gain map on the hair driven by the rendered irradiance, so the band dims when the light
> leaves and brightens when it returns while staying exactly where the photograph put it. It
> cannot MOVE the band, so row 5 stays formally unmet, but it removes no form. That is the one
> idea round 45 did not try, and it is cheap.
>
> **THE LIP GLOSS IS A DIFFERENT PROBLEM AND IS PROBABLY THE BETTER TARGET.** It is the other
> surface he named (round 44 §5: sheen 2.1 at the dark stations against 1.4 at the bright, corr
> −0.39; the face-delight separation takes only 9–15 % off). Three reasons it is not the hair:
> the lip is a SURFACE with a normal, so `--specular-model physical` applies directly and no
> fibre model or tangent estimate is needed; the vermilion has a DiffusionRenderer basecolor to
> use as a diffuse reference, which hair does not; and the band is 2–4 px, so far less
> photographed material is at risk. `hair_sheen.py`'s *shape* still applies — an envelope over a
> region-wide baseline, filtered in time in a head-stabilised frame, subtracted along each
> pixel's own chroma — and §2 and §4 are the bug list for building it. **But put an
> attenuation-only version in front of him FIRST, before any replacement.** That is this round's
> lesson and it costs one render to obey.
>
> **Reusable whatever happens to the hair:** `hair_region.py` measures a strand tangent and its
> confidence from any plate (coherence 0.70 over her hair, 0.79 in the band); `audit_r45_hair.py`'s
> azimuth test is a general "does this highlight follow the light" probe, not a hair-specific one;
> and §3's finding about round 44's statistic applies to ANY stage that subtracts energy from a
> region the composite can floor.
>
> **Do not re-derive, and do not re-open:** the crown band is 23× the dark hair at the SAME
> chroma, so no dichromatic split can touch it (round 44 §5); `face_delight --regions hair` does
> not move the audit; UnReflectAnything was ruled out on skin in round 43; render settings are
> not a lever for this void scene (round 44 §4c); the single-lobe physical fit explains only a
> fifth of the band (§6); and every bug in §2 and §4 is fixed, so a re-attempt would start from a
> working removal rather than from scratch.
>
> **Still open and still needing HIS decision, unchanged since round 44:** the fill panel
> reflecting in a shadowed eye is the RIG parking the fill BEHIND her face plane at the side
> stations (round44 §6) — fixing it re-renders the tour, both light passes AND the accepted eye
> pass, so ask. And porting `--only-light` to Blender's native LIGHT GROUPS halves the render
> cost; gate that port on reproducing `--only-light`, not on bit exactness (round44 §4c).
>
> **Three lessons from round 45.** First, and the hard one: **a metric handed down in a starter
> prompt can be satisfied by a rejected picture.** Round 44 wrote "round 45 must drive it
> POSITIVE"; it did, and he preferred the source. Put the picture in front of him before
> optimising the number, and treat an inherited numeric target as a hypothesis rather than a
> definition of done. Second: **a metric can be invalidated by the fix it is judging** — round
> 44's p95/p50 sheen ratio became a ratio of floor values the moment the band was subtracted
> (15 000 at the dark stations), so when a number moves oddly after a change, check its
> denominator before believing it. Third: **when an added term crawls, decompose before
> filtering.** The tangent field was filtered in time at real cost and moved the crawl by 0.011
> stops; the whole of it was the strand-texture exponent, which one ablation found.
>
> Keep: GENERIC tool, GPU first, plate pixels outside the pass exact, p95 AND max for anything
> temporal, a labelled crop at the scale of the defect in front of him before a full run, and a
> video before a verdict.
