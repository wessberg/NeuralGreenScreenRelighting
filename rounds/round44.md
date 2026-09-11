# Round 44 handoff — ACCEPTED (`r44_m50_s05`). Meso-shading from the neural normals is the lever for "plastic"; a specular on those normals is a Barbie doll unless it is scaled to the plate, broadened and broken up by her own pores

**Date:** 2026-09-11. Starts from round 43 (`r43_nospec`; the eye baseline `r42_wet` is
still the accepted one). The starter prompt asked for two things in order: a physically
scaled specular, then meso-shading from the neural normals under the new light. Built both,
in that order, and the order was wrong in the artist's eyes: the specular alone (frame 73,
first sheet) got *"Jesus christ this looks awful ... Now THAT's plastic"* and, after the
first correction, *"My complaint was that it looked a little plastic, meaning, the skin
lacked depth, imperfections. Now, with the added specular, it looks even more plastic. Now
it looks like a shiny barbie doll."* The meso-shading term on its own reads as a face under
a hard side light (§3). With the specular scaled to the plate's exposure, broadened to the
roughness real skin measures, and multiplied by the plate's own pore-scale texture, he then
said of the `r44_meso_spec` sheet: *"OK, yeah, I see an improvement in r44_spec_73_30 in the
meso + spec frames."*

**VERDICT: ACCEPTED.** After two rounds of dialling (§2b, §2c) he picked
`--meso-strength 0.5 --specular 0.5` and, on the videos:
*"Improved and accepted. It still doesn't look entirely 'human', but is better than before."*
**`r44_m50_s05` is the new baseline composite**, superseding `r42_wet` (which stays the
accepted EYE work that it is built on). Note the second half of his sentence: the
photorealism bar is NOT met and nothing here should be reported as meeting it.

Mid-round he also raised three things that are recorded below: lips and hair still carry the
capture room's shine (§5, and he raised the hair again after accepting), and the fill panel's
reflection in a shadowed eye (§6).

Videos: **`final/r44_m50_s05_compare.mp4` (R42 left, the accepted run right) and
`final/r44_m50_s05_panel.mp4` (face scale) are the accepted pair.** Also
`final/r44_meso_spec_compare.mp4`, `final/r44_meso_compare.mp4` (meso only, no specular),
`final/r44_spec_isolate.mp4` (meso vs meso+specular, which isolates the specular) and
`final/r44_spec_levels_panel.mp4`. Sheets in `composite/panels/r44_*.png`.

---

## 0. What was built

1. **Per-source irradiance passes.** `stress_proxy.py --ratio --only-light bulb|fill`
   renders the tour with ONE source (the other and the ambient trace at zero) and writes the
   world positions of bulb, fill and aim point into `render_manifest.json` /
   `logs/stress_proxy_<light>.json`. Transport is linear, so the two passes sum to the full
   pass: measured against `ratio_render` on the matte core, rel mean abs **≤ 0.012 on every
   frame** (`composite_manifest.json → light_pass_check`). `composite_ratio.py --light-passes
   bulb=r44_light_bulb,fill=r44_light_fill` uses the sum as E_target from then on. 3.5 s per
   frame per pass at 128 samples.
2. **Per-pixel light geometry** (`light_geometry.py`): the proxy's front shell is a
   back-projected pixel grid, so its vertices scattered by UV give a world-position map
   under every plate pixel (filled at quarter resolution, smoothed over the stride). From it,
   the unit vector to each source and to the camera, and the source distance, per pixel. A
   bulb 0.73 m from a 20 cm face is not a distant light: the direction to it swings ~15°
   across the face.
3. **Meso-shading** (`meso_specular.meso_modulate`, `composite_ratio.py --meso-sigma 0`):
   per source k, `E_k' = E_k × c / lowpass_σ(c)` with `c = max(n_neural · l_k(pixel), 0)`
   (the fill's cosine wrapped by the sine of its angular radius, ~1 at 0.44 m). Above σ the
   proxy stands (occlusion, falloff, solid angle); below it the neural normals' relief
   replaces the proxy's. σ is **measured**: the scale at which the proxy's rendered relief
   first equals the neural Lambert's, on the skin core of the anchor frame — **14.0 px** here
   (table in `composite_manifest.json → meso_shading.crossover_table`: the Lambert has 3.8×
   the proxy's relief at 2 px, 1.8× at 8 px, 0.8× at 16 px). Modulation on skin: bulb p95
   1.37 (max over frames 1.79), fill max 3.65 (clamp 4, never hit on the bulb). E_source is
   already divided at full resolution (`--source-detail-sigma` has been off since round 5),
   so "the same scale" needed nothing.
4. **Physical specular** (`meso_specular.ggx_specular`, `--specular-model physical`):
   `F(v·h) · D_GGX(n·h) · V_Smith(n·l, n·v) × π E_k'` per source, F0 0.04, the lobe widened
   by `r / (2 d)` for the source's angular size (a 60 cm panel at 0.45 m becomes a broad
   sheen, a 4 cm bulb a spot, from one roughness). Three corrections it took (§2):
   `--skin-albedo 0.40` (ASSUMED) scales it by the plate's measured skin albedo 0.241 / 0.40
   = 0.60; `--roughness-skin 0.5` (0.30 was typed; DiffusionRenderer measured 0.38 on her,
   round 43); `--specular-texture 2` multiplies it by `clip(plate / lowpass_2.5px(plate),
   0, 2)²` — a pore is a cavity whose facets face away, so it does not shine (HEURISTIC, and
   labelled so). `--specular-clamp 8` replaces the fixed 3× ceiling.
5. **Tools.** `probe_r44_relief.py` (the round-43 relief table, from the repo, with the
   composites as columns), `audit_r44_temporal.py` (head-stabilised step, and the 12 px
   high-passed step that the beat pulse cannot reach), `audit_r44_sheen.py` (lips and hair
   crown: does the shine track the new light?), `review_r44_face.py` (labelled face-scale
   stills, several runs side by side, rows per frame).
6. **Lips and hair in the shine separation** (`face_delight.py --stage separate --regions
   lips,hair --separate-name separate_lh`, `composite_ratio.py --face-delight-sub`): the
   paint footprint takes the vermilion and the hair crown; the shading fit stays on skin.
   Partial at best (§5).

Defaults are unchanged: every piece is a flag. The runs:

| run | what |
|---|---|
| `r44_meso` | light passes + meso-shading, `--specular 0` |
| `r44_meso_spec` | + physical specular, roughness 0.5, texture², albedo scale, clamp 8 — **the candidate** |
| `r44_meso_lh` | `r44_meso` + lips/hair shine subtracted |
| `r44_meso_spec_lo` | as `r44_meso_spec` with F0 0.028 (measured skin, IOR 1.4) and roughness 0.55 |
| `r44_meso_spec_half` | `_lo` with `--specular 0.6` — his "turn it down a little more" (§2b) |
| **`r44_m50_s05`** | **`--meso-strength 0.5 --specular 0.5` — THE ARTIST'S PICK (§2c)** |
| `r44_smoke*` | frame 73 only, bulb pass only — the sheets he rejected; not results |

## 1. Measured, before anything was built

Std of the 12 px high-passed log2 luminance on the eroded skin core
(`logs/r44_relief.json`; round 43's numbers used a different mask, the pattern is the same):

| frame | station | plate | E_source | E_target | Lambert(neural, per-pixel l) | r43_nospec | **r44_meso** | r44_meso_spec |
|---|---|---|---|---|---|---|---|---|
| 30 | below chin | 0.551 | 0.210 | 0.603 | 1.024 | 0.776 | **0.945** | 0.935 |
| 73 | camera right | 0.564 | 0.220 | 0.385 | 1.425 | 0.811 | **0.958** | 0.964 |
| 97 | behind rim | 0.560 | 0.216 | 0.234 | 1.041 | 0.652 | 0.663 | 0.673 |
| 120 | key off, back | 0.568 | 0.215 | 0.609 | 1.320 | 1.002 | 1.049 | 1.064 |

Two things this table says that round 43 did not: **E_source's own relief is 0.21** — the
SH-L2 of a softbox on these normals is smooth, so most of the plate's 0.56 is texture
(pores, brows, lips), not shading — and the composite's relief was already above the
plate's (0.81 at 73) because the ratio adds E_target's. The "Lambert column" is not a
reachable gate: its log relief is dominated by the terminator, where `cos → 0`. The
meso-shading raises the composite's relief by 18 % at the side and under-chin stations and
by 2–5 % where the bulb is behind her. Whether that is the right amount is his eye's call.

## 2. The specular, and the three things wrong with the first one

The first sheet (`r44_smoke_73.png`: bulb pass only, roughness 0.30, scene units) painted a
smooth bright band along the cheekbone, jaw and neck. Measured on frame 73: `f_r` peak 0.34
on skin, `L_s / (0.4 E)` p99 0.81 — and that estimate was itself 2.5× too kind, because the
plate's skin sits at **plate / E_source = 0.241** (stable to ±0.01 across the tour) against
a real skin albedo of ~0.40: the ratio composite's diffuse is `ρ_plate × anchor × E`, the
specular was `anchor × L_s`, so the highlight stood against a diffuse 40 % darker than
physics assumes. Corrections, in order, each shown on frame 73:

| variant | `f_r` max on skin | spec energy / skin luminance (f73) | his words |
|---|---|---|---|
| r0.30, scene units, no meso | 0.34 | 11.7 % | *"Now THAT's plastic"* |
| r0.38 × albedo scale 0.60 | 0.16 | 7.8 % | *"shiny barbie doll"* |
| r0.50 × 0.60 × texture², with meso (`r44_meso_spec`) | 0.06 (bulb) / 0.09 (fill) | 12.1 % (†) | *"I see an improvement"* |

(†) the meso term feeds the specular, and the fill's broad lobe is included; over the tour
the fraction is p50 0.117, p95 0.406, max 0.588 — the "front close" station at 0.5 m puts a
sheen over the whole face. Watch those frames in the video before trusting the number.

**The round-43 validation criteria for the specular are NOT met**, and that is worth
stating plainly. `audit_r43_shine.py` (nose bridge, p95/p50): r43 1.289, r44_meso 1.319,
r44_meso_spec 1.413 — the sheen rises, but its correlation with log E stays at 0.00 and the
dark-station value rises with the bright one (1.34 / 1.40), because the lobe is now broad
enough to lift the box's median as much as its p95. The peak of the added term at frame 73
lands at (847, 823) — the neck/hand under the fill's grazing Fresnel — not on the nose
bridge. What he saw as an improvement is not what that audit measures.

**A smooth specular on the neural normals is a doll by construction.** The normals resolve
~2.5 plate px; a pore is 0.2–0.5 px. Any lobe driven by them is spatially smooth over a
cheek, and a smooth highlight is what plastic has. The texture multiplication is the only
thing here that breaks it up, and it is a proxy for micro-normals nobody has measured.
Round 5's separation warning applies in reverse: the plate's texture is albedo AND softbox
shading; using it as micro-geometry double-counts the albedo.

## 2b. "Turn down the specular a little more still ... a little like an NPC"

His words on `r44_meso_spec`, after the sheet. Two steps, both shown on frames 73 / 30 / 10
(`composite/panels/r44_spec_levels_73_30_10.png`, video `final/r44_spec_levels_panel.mp4`):

| variant | F0 skin | roughness | gain | spec energy / skin luminance, p50 / p95 / max |
|---|---|---|---|---|
| `r44_meso_spec` | 0.040 | 0.50 | 1.0 | 0.117 / 0.406 / 0.588 |
| `r44_meso_spec_lo` | 0.028 | 0.55 | 1.0 | 0.094 / 0.379 / 0.571 |
| `r44_meso_spec_half` | 0.028 | 0.55 | 0.6 | 0.059 / 0.269 / 0.459 |

F0 0.028 is measured skin (IOR 1.4, Donner & Jensen); it takes 30 % off at normal incidence
but only 20 % off the energy, because past ~60° Schlick's Fresnel climbs to 1 whatever F0
is — the fill's broad grazing lobe and the neck/jaw rim dominate the total. The gain is not
physics; it is the artist's request, and `--specular 0.6` is how it is recorded. The p95/max
frames are the "front close" station (bulb 0.5 m in front): the whole face sheens there in
every variant. Video for his verdict: `final/r44_meso_spec_half_compare.mp4` (R42 left).

## 2c. It was the MESO, not the specular — and his setting is 0.5 / 0.5

He asked twice for the specular to come down and then said the reduction did not read. Both
were right, and the reason is that the specular was never the big term. Decomposed at frame
133 (the frame he screenshotted), on the eroded skin core:

| term | size |
|---|---|
| meso-shading against R42 | 0.17 stops p50, **1.75 stops p95** |
| low specular on top of the meso | +3.2 % p50, +17.8 % p95 |

`composite/panels/r44_what_changed_133.png` settles it: the middle tile is MESO ONLY, no
specular at all, and it already has the game-character look he named. His words: *"It might
actually be the meso, and not the specular, I'm referring to here. On the 'what changed'
image, 'MESO only' is the one that comes off as too strong."*

`--meso-strength` is the dial: an exponent on the modulation, so it scales the relief in
STOPS and leaves 1 (no change) fixed — 0 is the proxy alone, 1 the neural normals in full.
Mid-scale relief on skin (std of the 12 px high-passed log2 luminance), and the flicker cost:

| | relief f73 | relief f30 | hp12 median frame-p95 (stops) |
|---|---|---|---|
| plate | 0.564 | 0.551 | 0.142 (the control) |
| R42 accepted | 0.809 | 0.773 | 0.312 |
| **meso 0.5 (his pick)** | **0.867** | **0.842** | **0.338** |
| meso 0.8 | 0.921 | 0.899 | — |
| meso 1.0 | 0.959 | 0.937 | 0.386 |

**Half strength costs most of the flicker too**: the 12 px high-passed step is +8 % over R42
at 0.5 against +23 % at 1.0. The temporal worry in §3 is much smaller at his setting, though
still present (max frame-p95 2.05 vs R42's 1.50).

**`r44_m50_s05` is the round's result and it is ACCEPTED**, full 150 frames, video
`final/r44_m50_s05_compare.mp4` (R42 left) and `final/r44_m50_s05_panel.mp4`. His words on
the stills: *"Yeah, meso 0.5, spec 0.5 is the one."* On the videos: *"Improved and accepted.
It still doesn't look entirely 'human', but is better than before."*

## 3. The meso-shading, seen

`composite/panels/r44_meso_73_30.png` and `r44_spec_73_30.png`. At 73 (bulb hard camera
right) the nose casts a defined shadow onto the far cheek, the terminator runs down the nose
and the brow ridge instead of fading, the shadow-side socket reads as a socket; at 30 (bulb
under the chin) the nose shadow climbs the bridge and the cheeks roll off. `r44_smoke_meso`
vs `r44_smoke_none` at 3× (`r44_smoke_meso_zoom_73.png`) shows no blotching at the scale
of the normals' noise, but **I have not stepped all 150 frames at that zoom** (round 36's
rule), and the temporal audit says the term does move:

| run (`logs/r44_temporal.json`, stops, head-stabilised, skin core) | median frame-p95 | hp12 median | hp12 p95 | hp12 max (frame) |
|---|---|---|---|---|
| plate (control) | 0.135 | 0.142 | 0.604 | 0.771 (54) |
| E_target (the tour moves) | 0.356 | 0.208 | 0.834 | 1.309 (53) |
| r43_nospec | 0.447 | 0.313 | 1.135 | 1.507 (53) |
| r44_meso | 0.630 | 0.386 | 1.750 | 2.587 (53) |
| r44_meso_spec | 0.642 | 0.380 | 1.787 | 2.598 (53) |

The 12 px high-passed step (the band the beat pulse and the moving bulb cannot reach) rises
**23 % at the median and 54 % at p95** over r43. Frames 52–60 are her fastest head move
(the registration residual is in the plate's row too). The normals crawl ~2°/frame
(constraint row 14) and the meso term multiplies that crawl into the picture at 14 px and
below; nothing filters them in time yet. **If he sees crawl, that is where it comes from**,
and a temporal filter on the normals in a stabilised frame (round 37's edge-shading
treatment: symmetric window, seconds not frames) is the fix, not a lower `--meso-max`.

## 4. Per-source passes, the fill's cosine, and the anchor

* The fill is a 60 cm panel 0.44 m from her face: its angular radius is ~50°, so
  `meso_modulate` wraps its cosine (`wrap` = sin of the angular radius per frame, 1.0 here)
  — a Lambert to the panel's centre would put a terminator on her that a source this large
  does not cast. Its modulation still reaches 3.65 at p-max on some frames (`--meso-max 4`):
  look at the fill-lit side at 52–85 and 127 for a false edge.
* `--only-light` zeroes the world ambient trace with the fill (it belongs to the
  environment, not the bulb). "Key off" stations (fill_scale 0) have neither.
* The exposure anchor is unchanged by construction: 10.22 → 10.22 (median ratio on frame 0).

## 4b. Is the Cycles lighting production grade? The sampler is not the limit; the SCENE is

He asked mid-round whether the shadows and lighting are path traced at full fidelity or are
prototype lighting, and whether Blender has production-grade lighting exports we are not
using. Probed and measured (`composite/r44_conv_{128,1024,4096}`, one frame of the bulb pass).

**It is genuine path tracing.** Cycles through OPTIX on the GPU, ray-traced shadows,
random-walk subsurface, light tree on, caustics on, OpenImageDenoise with albedo+normal
inputs and the ACCURATE prefilter, adaptive sampling on at threshold 0.01. `stress_proxy.py`
overrides exactly three factory settings — device, `samples`, `use_denoising` — so everything
else is the Blender 5.2.1 default: 12 total bounces, 4 diffuse, 4 glossy, 12 transmission,
**0 volume**, indirect clamp 10, `BLACKMAN_HARRIS` filter at width 1.5.

**128 samples is already converged for this purpose, measured.** Frame 73 of the bulb pass
rendered at 128 / 1024 / 4096, compared as |delta log2| of irradiance luminance on the skin:

| pair | all skin p50 / p95 | the LIT half p50 / p95 / max |
|---|---|---|
| 128 vs 4096 | 0.018 / 0.513 | **0.011 / 0.045 / 0.310** |
| 1024 vs 4096 | 0.017 / 0.513 | 0.011 / 0.042 / 0.183 |
| 128 vs 1024 | 0.017 / 0.569 | 0.011 / 0.043 / 0.213 |

The all-skin p95 of 0.51 is NOT sampling noise — 1024 shows the same figure as 128, so it is
the floor of the shadow side, where the irradiance is ~0 and log2 of a near-zero is unstable.
On the half of the face the bulb actually reaches, 128 is within 0.045 stops of 4096 at p95.
**Do not spend a round raising samples.**

**What IS the fidelity limit, in order:**

1. **The scene is a black void.** The tour is a 4 cm emissive sphere plus a 60 cm emissive
   panel with a world background at 0.012 — no walls, no floor, no ceiling. So there is
   nothing to bounce off, `diffuse_bounces = 4` costs nothing and buys nothing, and the
   shadow side gets only the fill panel. In a real room most of the shadow-side fill IS
   bounce. The delivery path for this is the environment stages that already exist
   (`build_hdri_scene.py`, `bake_light_probes.py`, `scene_library/`); the bulb tour is a
   stress demo and was never meant to be a room.
2. **The proxy.** A displaced silhouette shell on a stride-2 grid with a flat back cap, no
   eye sockets, no separate hair, nothing behind her. Every shadow in the composite is cast
   by that shell (constraint rows 1-3).
3. `volume_bounces = 0`: no atmosphere, no haze, no visible beam.
4. The EXR passes are **half float** (`color_depth = '16'`); 32 is available.

**Production features Blender has that we are NOT using, one of which we hand-rolled:**

* **LIGHT GROUPS** (`view_layer.lightgroups`, `object.lightgroup`, `world.lightgroup` — all
  confirmed present in 5.2.1). Cycles writes one pass PER LIGHT GROUP from a SINGLE render.
  Round 44's `--only-light bulb|fill` renders the tour twice with one emitter zeroed each
  time to get the same thing. Light groups would halve that cost and make the passes sum
  EXACTLY instead of to within 1.2 %. **Round 45 should port `--only-light` to light groups**
  and keep `--only-light` as the reference the port is gated against.
* Cryptomatte (object / material / asset) and custom AOVs, both available on the view layer.
* 32-bit float EXR and the DWAA/HTJ2K codecs.

## 5. Lips and hair: measured his complaint, built a partial, not a fix

`audit_r44_sheen.py` on `r42_wet` (`logs/r44_sheen_r43.json`): lip vermilion sheen p95/p50
**2.11 at the dark stations vs 1.43 at the bright** (corr with log E −0.39); hair crown
**10.9 median, −0.39**; the brightest tenth of each region does not move between the
darkest and brightest quintile of the tour (offset spread 0.07 / 0.17 IPD). That is the
signature of the capture room's softbox: the gloss and the crown sheen scale with the ratio
like albedo and never follow the bulb. He is right.

`--regions lips,hair` in the separation takes, on the brightest decile, 9–15 % off the lips
and 15–18 % off the hair crown (`--smooth 0.006` for the finer band: `separate_lh2`), and
the sheen audit does not move (lips 2.03 → 2.10, hair 10.8 → 11.4 on the r44 runs: p50 and
p95 drop together). **The lip gloss is a 2–4 px saturated band and the hair sheen is
anisotropic; a term low-passed at 0.02 IPD with a skin-median floor was built for a broad
2 % sheen on skin and cannot take either.**

**And on the hair the DICHROMATIC route is dead, measured.** He raised the crown again after
accepting §2c (*"we still have the shine in the very top of her hair. Very clearly from a
light hanging above her head"*). On the crown region, over the tour:

| | luminance | chroma angle to the measured plate illuminant |
|---|---|---|
| the bright band (top 8 %) | 0.235 | 8.0–9.0° |
| the dark hair beside it (p20–p50) | 0.010 | 7.4–8.9° |

**The band is 23× the dark hair and they sit at the SAME chroma.** Her hair's own diffuse
colour is as close to the room's illuminant as the specular is, so `I = a·b̂ + s·ĝ` is
degenerate there — this is round 5's near-neutral case, and no dichromatic split (ours,
UnReflectAnything, or another) can separate it. What CAN be used is exactly that 23× ratio:
on near-black hair a pixel at 0.235 is at least 95 % specular, so the band is identifiable by
level alone.

But identifying it is not the job. **The crown sheen carries essentially all of the hair's
visible form** — delete it and the hair is a black mass. The construction round 45 needs is
remove-AND-replace: take the room's band off by level against a local dark-hair baseline
(strand texture preserved, footprint held in time, as `sclera_glint_removal.py` does for the
eye), then add an anisotropic hair specular (Kajiya-Kay or Marschner) from the tour's own
sources along a tracked strand direction. The lips need the band itself (no low-pass, the
DiffusionRenderer vermilion as the diffuse reference, and a check that the a-term is not
eating lip colour). Both are outside the skin mask, the face oval and the ratio's
per-material weights.

## 6. The fill panel in a shadowed eye (his screenshot, 13:22)

The pink crescent in her right eye at frame 73 is the fill panel reflected in the corneal
pass, and in the sheet he looked at it was doubly wrong: the smoke test had the face lit by
the bulb pass ALONE while the round-42 eye pass still carried the fill's reflection. In the
full runs the fill lights her too. Underneath is the round-42 caveat, now with numbers: the
rig parks the fill at `target − 0.6 × offset`, which at the camera-left/right stations is
**behind her face plane** (frame 73: fill at y = +0.18 m, her cornea at −0.01), so the
cheek receives it at grazing incidence and stays dark while the corneal mirror sees the
whole panel. From the right eye the panel's centre is **107° off the head axis** and its
edges span ~72–142°; the socket cup cuts at 100° temporal, so a third of it gets through,
and a mirror sphere images a source at 107° at 53° off its own axis — 6.2 mm from the
corneal axis, exactly the limbus. That is consistent physics for this rig; a real fill would
not be placed there. The fix is in the rig (keep the fill in front of the face plane, e.g.
clamp its y ≤ −0.3 m), and it re-renders the tour, both light passes and the eye pass. Not
done: it changes the accepted `r42_wet`, and that is his call.

## 7. Reproduce

```powershell
$P = "dependencies/CorridorKey/.venv/Scripts/python.exe"
$B = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
$S = "music_videos/reporter_test/shots/reporter_a"
& $B -b --factory-startup --python scripts/stress_proxy.py -- --shot $S --ratio --length 150 --samples 128 --only-light bulb --name r44_light_bulb
& $B -b --factory-startup --python scripts/stress_proxy.py -- --shot $S --ratio --length 150 --samples 128 --only-light fill --name r44_light_fill
$C = "--shot $S --eye-removal r42_wet_removal --corneal-dir r42_eye_pass --corneal-blur 0.35 --corneal-matte lids --light-passes bulb=r44_light_bulb,fill=r44_light_fill --meso-sigma 0"
& $P -B scripts/composite_ratio.py $C.Split() --name r44_meso --specular 0
& $P -B scripts/composite_ratio.py $C.Split() --name r44_meso_spec --specular-model physical --roughness-skin 0.5 --specular-texture 2 --specular-clamp 8
& $P -B scripts/composite_ratio.py $C.Split() --name r44_m50_s05 --meso-strength 0.5 --specular-model physical --roughness-skin 0.55 --f0-skin 0.028 --specular 0.5 --specular-texture 2 --specular-clamp 8   # THE PICK
& $P -B scripts/face_delight.py --shot $S --name r43_delight --stage separate --regions lips,hair --separate-name separate_lh
& $P -B scripts/composite_ratio.py $C.Split() --name r44_meso_lh --specular 0 --face-delight r43_delight --face-delight-sub separate_lh
& $P -B scripts/probe_r44_relief.py --shot $S --frames 30,73,97,120 --dirs r43_nospec,r44_meso,r44_meso_spec
& $P -B scripts/audit_r44_temporal.py --shot $S --dirs r43_nospec,r44_meso,r44_meso_spec,r44_meso_lh
& $P -B scripts/audit_r43_shine.py --shot $S --dirs r43_nospec,r44_meso,r44_meso_spec --out r44_shine_audit.json
& $P -B scripts/audit_r44_sheen.py --shot $S --dirs r44_meso,r44_meso_lh --out r44_sheen_lh.json
& $P -B scripts/review_r44_face.py --shot $S --dirs r43_nospec,r44_meso,r44_meso_spec --labels "R43 nospec|R44 meso|R44 meso + spec r.50 tex" --frames 73,30 --half-w 1.0 --half-h 1.2 --out r44_spec_73_30.png
& $P -B scripts/review_video.py --shot $S --name r44_meso_spec --compare r42_wet --label "R44: meso-shading + specular" --compare-label "R42: wet eye (accepted)" --out r44_meso_spec_compare.mp4
& $P -B scripts/review_r37_panel.py --shot $S --name r44_panel --dirs "r42_wet/.,r44_meso/.,r44_meso_spec/." --labels "R42|R44 meso|R44 meso+spec"
```

## 8. Starter prompt for round 45

> **Round 44 built the meso-shading from the neural normals and a physical specular per
> light source, and `r44_m50_s05` (`--meso-strength 0.5 --specular 0.5`) is ACCEPTED:
> *"Improved and accepted. It still doesn't look entirely 'human', but is better than
> before."* It is the new baseline composite and the thing to build the next comparison
> against. Read round44.md §2c, §5, §6 first.** He says it is still not human, so do not
> report the bar as met. Crawl is not yet ruled out by his eye: the 12 px high-passed step is
> +8 % over R42 at his setting (it was +23 % at full meso), so if he ever reports it, filter
> the neural normals in time in a head-stabilised frame (symmetric window in SECONDS, as
> round 37's edge shading), never the modulation. **The two hard lessons of this round, do
> not relearn them:** "plastic" means
> missing depth and imperfections, so the lever is SHADING RELIEF and a smooth specular makes
> it worse; and when he says an effect is too strong, DECOMPOSE before you tune — he asked
> twice for the specular to come down when the term he was reacting to was the meso, 1.75
> stops against the specular's 0.18.
>
> **Then the hair, which he raised the moment the skin landed** (*"we still have the shine in
> the very top of her hair. Very clearly from a light hanging above her head"*). §5 has the
> measurement: the crown band is 23× the dark hair beside it and at the SAME chroma, so no
> dichromatic split can touch it, but on near-black hair a level test is enough to find it.
> Build remove-AND-replace — the room's band off against a local dark-hair baseline with the
> strand texture kept and the footprint held in time (`sclera_glint_removal.py` is the
> pattern), then an anisotropic hair specular (Kajiya-Kay/Marschner) from the tour's own
> sources along a tracked strand direction. Removing without replacing turns her hair into a
> black mass: that band carries nearly all of its form. The lip gloss is the same family and
> smaller (§5). And the fill panel reflecting in a shadowed eye is the RIG parking the fill
> behind her face plane at the side stations (§6) — fixing it re-renders the tour and the
> accepted eye pass; ask him. Keep: GENERIC tool, GPU first, plate pixels outside the passes
> exact, p95 and max for anything temporal, a face-scale sheet in front of him before a full
> run, a video before a verdict.
