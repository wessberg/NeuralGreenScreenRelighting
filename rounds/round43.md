# Round 43 handoff — the room's shine off her face: built, measured at 2 %, and it is not what he sees. What he sees is the SHADING.

**Date:** 2026-09-11. Starts from round 42 (`r42_wet`, accepted). The artist asked whether we
had PBR maps of her skin, because *"her skin is shining equally even as we reduce the light
or move it away"*, and proposed removing the capture room's shine with UnReflectAnything or a
similar model and applying our own specular, plus ML subsurface scattering. Mid-round, looking
at `composite/r43_nospec/000073.png`: *"still a kind of 'plastic skin' … there's something
off about her, and I'm thinking it might be the skin shading?"* — and: *"Does it look natural
to you? Is there anything we can do about our shading to make it more realistic?"*

**Answer, measured (§3): no, it does not look natural, and the skin shading is the reason —
but not the part of it the round set out to fix.** The old room's specular on her skin is
about 2 % of skin luminance and removing it changes nothing visible. The composite's OWN
specular is effectively absent (0.01–0.45 % of skin luminance, peak pixel 0.045), and the
rendered irradiance carries about HALF the meso-scale shading relief that the neural normals
say the new light should produce, while the plate keeps 100 % of the softbox's. Her face is
still modelled by the green-screen room's light; the bulb only paints a smooth colour gradient
over it. That is the plastic look.

---

## 0. What was built

1. **`face_delight.py`** — five stages on a square face crop (face-track bbox union, padded,
   571 px native here; mid-grey behind the matte as the normals stage does):
   `prepare` → `inverse` (DiffusionRenderer inverse SVD, basecolor/roughness/normal, 24-frame
   windows with a 6-frame cross-fade, 512 px, 463 s for 150 frames × 3 passes) →
   `unreflect` (UnReflectAnything's own highlight head + diffuse decoder, 88 s for 150) →
   `analyse` (both against the plate, in linear light, on skin) → `separate` (§1).
2. **`composite_ratio.py --face-delight <name>`** — subtracts the separated shine
   (`separate/S/*.png`, 16-bit linear) inside its weight before the ratio; exact fg outside.
   Manifest records it; per-frame `face_delight_px` and energy removed.
3. **`audit_r43_shine.py`** — the acceptance test for the artist's complaint, read from the
   composites only: p95/p50 of skin luminance in the forehead, nose bridge, cheek and chin
   boxes ("sheen") against the rendered irradiance level per station; correlation of sheen
   with log E; sheen at the darkest vs brightest quintile of stations.

Runs: `composite/r43_delight` (models, separation, previews), `composite/r43_delight_comp`
(the composite with the shine subtracted, on the r42 eye passes), `composite/r43_nospec`
(same with `--specular 0`, the control). Videos `final/r43_delight_compare.mp4` (R42 left,
R43 right) and `final/r43_delight_panel.mp4`. **He will not see a difference; §3 says why.**

## 1. The separation, and the three things it took to make it stable

UnReflectAnything is out: its highlight head finds nothing on her face (the hl map is a faint
outline), and its "diffuse" is a whole-frame re-render with a colour shift, so plate − diffuse
is the whole face (16 % of skin luminance, uniform), not shine.

DiffusionRenderer's basecolor is a plausible flat albedo (the forehead sheen is largely gone
in it), but it is an ALBEDO with an arbitrary scale. To compare it with the plate it has to be
lit: `plate ≈ bc × E(n) + S`. Three things had to be true of `E(n)`:

* **Unregularised SH-L2 is ill-conditioned over a face.** Her normals span a narrow cone about
  the camera axis; the nine basis functions are near-collinear on that set and the fit
  returned coefficients of ±70 that cancelled to a shading of order 1 — and flipped sign
  between adjacent frames (DC term +70 → −24 over 24 frames; S/plate 0.008–0.123). Now SH-L1
  with a ridge on the directional terms (`--sh-order 1 --ridge 0.05`): S/plate 0.016–0.027
  over 150 frames, step max 0.003.
* **No function of the normal can express distance falloff.** The hand, nearer the softbox
  than the face, came out as one solid block of "shine". Now the stage is restricted to the
  mediapipe face oval (`--oval-grow`), and the shading carries a spatial quadratic falloff
  fitted alternately with the SH (`--no-falloff` to drop).
* **The plate is not `shot.resolution`.** That is the DELIVERY size (3840×2160 against a
  1920×1080 plate); using it cropped the normals from the wrong quadrant and put the S canvas
  at twice the plate. With the right normals the SH explains more and S fell from 5 % to 2 %.
  `plate_size()` reads the fg on disk. (A per-frame gain-fitted `bc × E_source` from
  `logs/source_light.json` was tried first: the residual was the whole lit side of the face,
  because the measured SH is one fit for the clip and the model's albedo scale is not ours.)

The residual `r = plate − D` is split as `r = a·b̂ + s·ĝ`, `a, s ≥ 0` — round 5's dichromatic
decomposition, but with `b̂` from the MODEL's diffuse at that pixel instead of a local opening
(the opening mixed materials across the mouth and the eye and destroyed teeth and sclera).
`ĝ` is the plate's illuminant from her corneal glints (`logs/specular_separation.json`,
1.0 / 0.946 / 0.917). `s` is low-passed at 0.02 IPD, the skin median subtracted as a floor,
feathered 0.06 IPD inside every mask edge, eyes and mouth interior cut out (landmark polygons
+ 0.06 IPD), SH and falloff median-filtered over 0.1 s. Footprint: forehead centre, nose
bridge, upper cheek, chin. `logs/r43_delight_separate.json`.

| separation (150 frames, skin core) | value |
|---|---|
| S / plate luminance | p50 0.022, max 0.027 |
| S frame-to-frame step | p95 0.006, max 0.078 |
| plate's own step (same pixels) | p95 0.044 |
| floor subtracted | p50 0.009 |

**The DiffusionRenderer roughness is flat on skin** (p10/p50/p90 0.35/0.39/0.44, median 0.38)
— a constant, like the one we already use (0.30). It was not wired in; a roughness map from a
single fixed light is a prior, not a measurement.

## 2. Licences

DiffusionRenderer: NVIDIA licence §3.3, **non-commercial (research or evaluation) only**.
UnReflectAnything: code MIT, DINOv3 encoder under Meta's DINO licence. This round is an
evaluation, like round 31. Nothing from either model is in an output pixel: S is a
subtraction the model only REFERENCES.

## 3. Why he sees no change, and what he is actually seeing (`logs/r43_shine_audit.json`, `r43_shine_nospec.json`)

Forehead sheen (p95/p50 of skin luminance), per run, over the tour's 9.5-stop range of E_t:

| run | sheen p50 | dark stations | bright stations | corr(sheen, log E) |
|---|---|---|---|---|
| r42_wet | 1.147 | 1.201 | 1.151 | 0.006 |
| r43_delight_comp (shine off) | 1.146 | 1.157 | 1.144 | 0.031 |
| r43_nospec (shine off, `--specular 0`) | 1.146 | 1.157 | 1.144 | 0.031 |

Nose bridge, cheek and chin: the same to three decimals. So:

1. **The plate's residual shine is 2 % and invisible.** Its removal cannot be what fixes the
   complaint. (The 6× difference image at the brightest station shows one faint blob on the
   forehead centre.)
2. **The composite's additive specular is effectively OFF.** `--specular 0` changes the
   composite by 0.01–0.45 % of skin luminance (peak pixel 0.045 at frame 120). Measured why:
   `specular_term()` normalises the GGX lobe to peak 1 and then multiplies by Schlick Fresnel
   (0.04 at normal incidence), so the term tops out at 0.03 on skin at every station (max over
   skin 0.030–0.032 at frames 30/73/97). The physical peak of a roughness-0.3 lobe under a point
   source is F·D = 0.04 × 1/(π·0.09²) ≈ 1.6 × E — about ten times the diffuse. The
   normalisation threw the magnitude away and the docstring says the caller supplies it; the
   caller supplies `conc × E_t`, which is the diffuse scale. **A bulb hard-left of her face and
   no hot spot on the nose bridge or brow is what reads as wax.**
3. **The rendered irradiance has half the relief the normals say the light should produce.**
   Std of the 12 px high-passed log2 luminance on skin:

   | frame | plate (old light) | E_t (proxy render) | Lambert on the NEURAL normals, same direction |
   |---|---|---|---|
   | 30 | 0.743 | 0.624 | 0.966 |
   | 73 | 0.751 | 0.420 | 0.735 |
   | 97 | 0.752 | 0.298 | 0.742 |

   The plate keeps ALL of the softbox's meso-shading (the E_source division is band-limited to
   the measured crossover so it does not remove detail the proxy cannot put back — that is
   deliberate, round 4), and the proxy adds the new light's at 40–60 % strength. At frame 73
   the bulb is at (−0.97, 0.24, −0.01), hard camera-left: a real face has a nose shadow across
   the right cheek, a dark right socket, a cheek that rolls off. The composite has a smooth
   warm/cool split painted over a face whose nose, brow and cheeks are still lit from the
   front-top-left by the softbox. Two lights describe her form at once, and the eye reads the
   conflict as a doll.

**PBR maps are not the lever for this.** Albedo and roughness change what a renderer would do
with light; here the renderer's irradiance is used only as a ratio and the picture is still
the plate. The two levers are the specular's magnitude (item 2, a scaling bug) and the
meso-scale shading (item 3, a construction): shading from the NEURAL normals under the new
light with the old light's shading divided out at the SAME scale, the proxy supplying only
occlusion and the far field. `relight_sh.py` (stage 10, the gain map) is the normals-based
half of that and was measured to have a ceiling on SHADOWS, not on shading; the proxy is
the shadows half. They have never been combined.

## 3b. The hair edge he zoomed into (screenshots 2026-09-11 12:57)

Two crops of the review at frame 73 show a hard, stair-stepped hair silhouette with a
one-pixel dark fringe and no partial coverage on flyaways. That is the MATTE, not the light.
Measured on `key/alpha/000073.exr`: 98.5 % of all alpha pixels are exactly 0 or 1; along the
hair silhouette there are 1.13 fractional pixels per edge pixel (skin edge 1.47), i.e. a
one-pixel ramp. Strand-level alpha does not exist in this key, so under any relight the hair
sits on the background like a cut-out. This is constraints row 13 (edge integration) and row
11 (the plate is 4:2:0 and was keyed at 1080p); on his own footage the answer is the one
AGENTS.md already gives: all-intra transcode, chroma upsampled, key at native 6K and
downscale after. On this stock plate there is no native 6K to key. The hair's SHEEN also
still says softbox (round 5's first complaint); hair is outside the skin mask and outside
this round's face oval, and gets the proxy's ratio as a mass.

## 4. Subsurface scattering

Not a network's job. Cycles folds subsurface into the diffuse passes the ratio is built from
(trap 4), and the proxy runs random-walk skin SSS, so R already carries the terminator
softening and the red shift at a shadow edge in principle. Whether it is visible is bounded by
item 3: at 40–60 % relief there is no terminator to soften.

## 5. Still open — the round 44 list

1. **Specular magnitude** (`composite_ratio.specular_term`): a physically scaled lobe —
   `F · D · G / (4 n·l n·v)` with the un-normalised D — from the bulb's actual direction and
   solid angle (the render manifest has the station), gated by the rendered occlusion; the
   fill panel's broad lobe as a second term. Validate on the audit: sheen at bright stations
   must rise above dark ones, and a hot spot must sit on the nose bridge/brow at frame 73.
   Watch the clip at 8-bit: the master keeps the values.
2. **Meso-shading from the normals**: `E_t' = E_t,proxy × Lambert(n_neural, L_new) /
   lowpass(Lambert(n_neural, L_new))` with `L_new` the SH-L1 fit of E_t against the proxy
   normals (render a normal pass, or fit against the neural normals with the same ridge as
   §1) — and `E_source` divided at the SAME sigma, which means the band-limit comes off. Gate on
   the relief table in §3: E_t' should reach the Lambert column. Look at frame 73 first.
3. The 2 % shine removal stays available (`--face-delight r43_delight`) and costs nothing; it
   is not a fix and must not be presented as one. If items 1–2 land and the softbox sheen then
   shows through the new hot spot, revisit with `--floor-pct` lower than the median.
4. UnReflectAnything is not a tool for this plate. Do not re-evaluate it on skin.

## 6. Reproduce

```powershell
$P  = "dependencies/CorridorKey/.venv/Scripts/python.exe"
$PD = "dependencies/diffusion-runtime/.venv/Scripts/python.exe"
$PU = "dependencies/unreflect-runtime/.venv/Scripts/python.exe"
$S  = "music_videos/reporter_test/shots/reporter_a"
& $P  -B scripts/face_delight.py --shot $S --name r43_delight --stage prepare
& $PD -B scripts/face_delight.py --shot $S --name r43_delight --stage inverse
& $PU -B scripts/face_delight.py --shot $S --name r43_delight --stage unreflect
& $P  -B scripts/face_delight.py --shot $S --name r43_delight --stage analyse
& $P  -B scripts/face_delight.py --shot $S --name r43_delight --stage separate
& $P  -B scripts/composite_ratio.py --shot $S --name r43_delight_comp --eye-removal r42_wet_removal --corneal-dir r42_eye_pass --corneal-blur 0.35 --corneal-matte lids --face-delight r43_delight
& $P  -B scripts/composite_ratio.py --shot $S --name r43_nospec --eye-removal r42_wet_removal --corneal-dir r42_eye_pass --corneal-blur 0.35 --corneal-matte lids --face-delight r43_delight --specular 0
& $P  -B scripts/audit_r43_shine.py --shot $S --dirs r42_wet,r43_delight_comp --out r43_shine_audit.json
& $P  -B scripts/audit_r43_shine.py --shot $S --dirs r43_delight_comp,r43_nospec --out r43_shine_nospec.json
& $P  -B scripts/review_video.py --shot $S --name r43_delight_comp --compare r42_wet --label "R43: room shine off the face" --compare-label "R42: wet eye" --out r43_delight_compare.mp4
& $P  -B scripts/review_r37_panel.py --shot $S --name r43_delight_panel --dirs "r42_wet/.,r43_delight_comp/." --labels "R42|R43 shine off"
```

## 7. Starter prompt for round 44

> **Round 43 measured why her skin reads as plastic, and it is the SHADING, not the shine.
> Read round43.md §3 and §5 first.** The capture room's specular on her face is ~2 % of skin
> luminance (`face_delight.py`, DiffusionRenderer basecolor as the diffuse reference, an
> SH-L1 + falloff shading refit, the dichromatic split against her measured illuminant);
> removing it is invisible and the sheen audit (`audit_r43_shine.py`) does not move. Two
> measured defects remain: (1) the composite's additive specular is effectively OFF — the GGX
> lobe is normalised to peak 1 and then multiplied by Fresnel 0.04, so it tops out at 0.03
> everywhere; `--specular 0` changes the picture by under 0.5 %. (2) The proxy's irradiance
> has 40–60 % of the meso-shading relief that a Lambert of the NEURAL normals under the same
> light has (std of 12 px high-passed log2: 0.42 vs 0.74 at frame 73), while the plate keeps
> 100 % of the softbox's. Build, in this order: a physically scaled specular from the bulb's
> actual direction and solid angle (validate: sheen rises at bright stations; a hot spot on
> the nose bridge at frame 73); then meso-shading from the neural normals under the new light
> with `E_source` divided at the same scale, the proxy supplying occlusion only (validate:
> E_t' reaches the Lambert column of the relief table). Show frame 73 side by side with
> `r43_nospec` before anything else. Do not add PBR maps: the albedo and roughness are not
> what is missing. Keep: GENERIC tool, GPU first, plate pixels outside the passes exact, p95
> and max for anything temporal, a video before a verdict.
