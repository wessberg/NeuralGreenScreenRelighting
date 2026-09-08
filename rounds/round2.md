# Round 2 — neural normals, and the finding that they were not the bottleneck

**Date:** 2026-09-08 · **Working tree:** clean at commit time

Round 1 named directional modelling as the gap and neural normals as the fix. The normals were
built, validated and wired in. **They were not the fix**, and the measurement that says so is the
most useful thing in this round. The real ceiling is the ratio formulation itself, and lifting it
means ray-traced geometry — which now exists, and casts a nose shadow.

---

## 1. The headline: round 1's diagnosis was wrong, and here is the number

Round 1 said depth-gradient normals were "structurally flat", so the key sweeping 200° could not
move across the face. Measured on frame 75, driving a synthetic key across ±90° of azimuth and
tracking where the shading centroid lands inside the face box:

| normals | shading centroid travel |
|---|---|
| depth-gradient | 66.1 px = **60 %** of face width |
| Sapiens2 | 54.7 px = **50 %** of face width |

The depth normals were never flat. They are *noisy* — wider azimuth spread (46.5° vs 30.2°) but
incoherent — and both fields respond to a moving key about equally. Swapping them therefore could
not fix what round 1 attributed to them, and did not:

| metric | round 1 (depth) | round 2 (Sapiens2) |
|---|---|---|
| actor luma swing, relight ON | 118.0 % | **136.4 %** |
| actor luma swing, OFF (control) | 0.86 % | 0.86 % |
| **corr actor↔room luminance** | **+0.39** | **+0.295** ↓ |
| corr actor↔room warmth | +0.25 | **+0.295** ↑ |

**The number to beat was +0.39 and it was not beaten.** Level and warmth tracking improved,
directional correlation regressed. Reported as a miss, not dressed up.

Sapiens2 normals are kept anyway, and are the right input for the reasons that actually hold:
they are coherent instead of noisy, they extinguish correctly when lit from behind (a real face
lit from behind goes dark; the noisy field never fully does), they carry true facial structure —
and, decisively, **they turned out to be what makes the proxy geometry work** (§3).

**Why direction is really lost:** `out = plate × gain` can only redistribute light the plate
already shows. Measured across the shot, the gain map's left/right balance over the face moves
from 0.42 to 1.19 — real directional variation, but always smooth, always broad, and capped by
the 1.5-stop `tanh` at 2.83:1 total range. A key at 90° to a face produces a lit/shadow ratio far
beyond that, with a hard terminator. A smooth multiply cannot make a terminator. That is the
ceiling, and it is structural.

---

## 2. Sapiens2 normals: built, and validated the hard way

`scripts/neural_normals.py` + `scripts/validate_normals_convention.py`.

`facebook/sapiens2-normal-1b` runs on the existing torch 2.8 after upgrading transformers
5.2.0 → 5.16.1 in the pipeline venv, **with numpy pinned at 2.4.2** (the unpinned upgrade wanted
2.5.3). EXR reads re-verified intact afterwards, since every stage depends on them. 150 frames at
**0.42 s/frame**. Input is cropped to the matte bbox expanded to 3:4, composited onto mid-grey,
and converted linear → sRGB before ImageNet normalisation; all three matter.

### The convention was NOT what round 1 predicted

Round 1 predicted camera→world as `(x, −y, −z)` — a sign flip. Measured, it is
**`world = (c₀, −c₂, c₁)`**: an *axis swap*. Round 1's guess would have written the vertical axis
into the depth slot — lighting wrong in every direction, invisible in any single frame,
ungradeable afterwards.

Four independent tests, because one direction cannot catch a reflection:

| test | result |
|---|---|
| analytic sphere, all **48** signed permutations ranked against exact ground truth | best `+x −y +z` at **6.09°**, runner-up **43.96°**, margin **37.9°** |
| horizontal flip (in distribution — the config ships `flip_pairs`) | negates image x ✓ |
| vertical flip (out of distribution, reported as weaker) | negates image y ✓ |
| anatomy: face-left, face-right, brow-vs-jaw, faces-camera | 4/4 pass |

Evidence in `logs/normals_convention.json`. `neural_normals.py` **refuses to run** without it.

**Licence correction:** round 1 recorded Sapiens2 as "commercially licensed". The model card says
*Sapiens2 License*, a custom Meta licence, not a standard permissive one. Read it before this
reaches a commercial release. The artist has already accepted NC terms while experimenting, so
this changes nothing today, but it must not harden into a fact.

---

## 3. The Cycles proxy stage — and it casts a nose shadow

`scripts/make_proxy_mesh.py`, `scripts/render_proxy.py`, `scripts/validate_proxy_shadows.py`.

Shading is `dot(n, l)`. Shadowing is "is the path to the light blocked?". No normal map answers
the second question, at any quality, because a cheek pixel's normal contains nothing about the
nose in front of it. Geometry does.

The proxy is a displaced silhouette shell: matte silhouette (accurate, already produced by the
key), displaced by metric depth using the *same* back-projection and calibration as the normals,
closed with a flat back cap and stitched side walls so it is opaque to rays arriving from behind.

**Two real defects found and fixed while building it:**

1. **Monocular depth over-estimates facial relief by 2.2×.** The raw fit produced a 0.50 × 0.68 ×
   0.53 m bust — a person 68 cm deep. Depth models get ordering right and metric relief badly.
   Fixed by anchoring the relief scale on the one region whose true depth is known: the face,
   set to 0.085 m of visible-surface depth (nose tip to cheek plane). Scale factor 0.455;
   result 0.50 × **0.31** × 0.53 m. An over-deep proxy throws over-long shadows.
2. **The depth shell alone renders as a mannequin** — no nose ridge, so an uplight threw no nose
   shadow. Fixed by **normal-integrated detail**: for a surface `Y = f(X,Z)` the normal *is* the
   gradient field, so integrating the Sapiens2 normals (Poisson, DCT, Neumann — exact and
   non-iterative) recovers the fine relief depth lacks. Depth supplies frequencies above 5 cm,
   the integrated normals below it, because their error profiles are opposite. Recovered relief
   amplitude p95 = **3.3 cm**. This is where the neural normals genuinely earn their place.

### Acceptance test — the artist's own suggestion, and it passes

A light under the chin, then one above the head. Occlusion is proven not by contrast but by the
shadow *moving to the other side of the nose*:

| | shadow centroid (0 = top of face) |
|---|---|
| uplight | 0.499 |
| toplight | 0.644 |
| **travel** | **0.145 of face height** |

`VERDICT PASS`, 5/5 checks, `logs/proxy_shadows.json`. Visible in the previews: uplight gives lit
nostrils and shadow thrown **up** across the eyes; toplight gives a nose shadow **down** onto the
lip, plus eye sockets, brow and a chin shadow on the neck.

Subsurface is set up with Blender 5.2.1's `RANDOM_WALK_SKIN` and Jensen et al. 2001 skin1 radii
(3.67 / 1.37 / 0.68 mm), and the **dedicated** `use_pass_subsurface_direct/indirect/color` are
enabled — a diffuse-only ratio silently drops all subsurface response.

**Not yet done: the composite path.** `out = plate × R_d × S + G` is not wired. The proxy renders
and casts correct shadows; nothing yet applies them to the plate. That is round 3's first job.

---

## 4. The artist's capture format, now recorded — `capture_format.json`

**Panasonic LUMIX S5 IIX, 6K 3:2 open gate 5952×3968, HEVC 10-bit 4:2:0 Long GOP 200 Mbps,
V-Log / V-Gamut · SIRUI Saturn 35 mm Full-Frame Anamorphic, 1.6× squeeze · 180° shutter at
24 fps · WB 5600 K · delivering 3840×1600 (2.40:1).** Given by the artist across this session.

`scripts/anamorphic.py` holds all the arithmetic so no factor of 1.6 is ever retyped. The
geometry works out exactly, which is worth stating because it looks like it should not:

* 3840×1600 desqueezed ÷ 1.6 = **1.50:1 squeezed**, and a full-frame gate is 36 × 24 mm = 1.50:1
  **exactly**. The full open gate is used, no crop.
* Horizontal equivalent focal 35 / 1.6 = **21.875 mm** → hFOV **78.90°**; vertical stays 35 mm →
  vFOV **37.85°**.
* Rendering the desqueezed frame at 21.875 mm on a 36 mm horizontal sensor reproduces **both**,
  because the implied vertical sensor is 15 mm and 15/21.875 = 24/35. No pixel-aspect trickery.

Defocus does not come free, in two ways. Amount: blur ∝ focal²/N, so N scales by 1/1.6² = 0.3906
(T2.9 → f/1.13) — this is why anamorphic gives long-lens depth of field at wide-lens coverage.
Shape: `dof.aperture_ratio`, and its direction was **measured, not assumed** — ratio 1.6 yields a
1.6:1 vertical oval, error 0.0 (`logs/anamorphic_validation.json`).

Wired into `build_hdri_scene.py` behind `camera.anamorphic` in `shot.json`, opt-in per shot: the
reporter plate is 16:9 spherical stock footage and must not inherit it.

**The camera spec confirms the geometry rather than merely permitting it.** 5952/3968 = 1.5000
exactly; desqueezed by 1.6 that is 2.400:1; delivery is 3840×1600 = 2.40:1. Camera mode, lens
squeeze and delivery frame agree with no crop. Desqueezed native is 9523×3968 against a
3840×1600 delivery — **2.48× linear oversampling**, the biggest quality asset in the chain.

### Recording mode: a real trade, evaluated rather than assumed

The artist asked whether 5.8K 17:9 ProRes 422 HQ would suit the pipeline better than 6K open gate.
Recorded in `capture_format.json`; the numbers:

| | 6K 3:2 open gate | 5.8K 17:9 |
|---|---|---|
| desqueezed aspect | **2.400:1 — the delivery aspect exactly** | 3.024:1 |
| crop for 2.40:1 | none | **79.4 % of width; 20.6 % discarded** |
| oversampling over 3840×1600 | **2.48×** | 1.91× |
| sensor used | 36.0 × 24.0 mm | 27.7 × 18.5 mm |
| hFOV at 35 mm | 78.9° | 64.7° |
| codec | HEVC 4:2:0 **Long GOP** 200 Mbps | **ProRes 422 HQ all-intra**, 1.6 Gbps, 24P native, LPCM |
| storage | 1.5 GB/min | 12 GB/min |

**Recommended: the ProRes mode**, because it deletes two of the three keying problems outright.
All-intra removes the Long GOP mosquito noise around the silhouette and every hair — the thing
that makes a matte crawl — and 4:2:2 keeps full vertical chroma resolution, which matters because
keying *is* a chroma operation. 1.91× is still comfortable supersampling, and the headroom being
given up existed mainly to fight compression artefacts in the first place. The cost is framing,
not quality: the 35 mm behaves like roughly a 44 mm.

Open question for the artist: **does the body offer open gate 3:2 in ProRes?** That would be
strictly better than either and the trade would disappear.

### Two new red rows this created

**V-Log / V-Gamut decode (constraint 18, red).** Every stage here assumes scene-linear Rec.709:
keyer thresholds, depth and normal estimators, the SH irradiance ratio, the PQ grade. Feed V-Log
in undecoded and all of them are wrong while still producing a picture. Decode on ingest, before
keying, and never apply a creative display LUT first — it is not invertible and discards the
highlight range the HDR grade exists for. The transfer and gamut constants must be **verified
against Panasonic documentation and a shot ColorChecker**, not inherited; they are recorded as
unverified in `capture_format.json`.

**4:2:0 Long GOP keying.** Chroma is stored at half resolution (2976×1984) and a green-screen key
is a chroma operation, so chroma must be upsampled before keying or every matte edge inherits a
blocky staircase. Long GOP puts mosquito noise around exactly the high-contrast edges that matter
— the silhouette and every hair — so transcode to an all-intra intermediate **before** keying or
the matte will crawl. And key at native 6K, downscaling afterwards: at 2.48× oversampling the
matte comes out effectively supersampled and hair edges get real fractional alpha instead of a
hard decision.

---

## 5. Editorial masters for Resolve — `scripts/master_export.py`

ProRes 4444 / 4444 XQ / DNxHR 444, 4:4:4, Rec.2020 ST2084 PQ at a 1000-nit mastering peak, from
the same EXR sequence and the same transform chain as the HEVC master.

**10-bit is the ceiling, and it is ffmpeg's, not the format's.** Verified against the bundled
build: `prores_ks` advertises `yuv422p10le yuv444p10le yuva444p10le`, `dnxhd` advertises
`yuv422p yuv422p10le yuv444p10le gbrp10le`. No 12-bit encoder exists in either.

**For more than 10 bits, do not use a video codec.** `intermediate/linear_4k/*.exr` is 16-bit
half float, scene-linear and unquantised, and Resolve imports EXR sequences natively — strictly
better than any ProRes. `--report` prints that path. The mov timescale is the frame-rate
numerator, per round 1's hard-won lesson.

**Exercised end to end against `proof_of_concept`, and probing the output caught a real bug the
command line hid.** Colour tagging does not survive the obvious flags:

| encoder settings | resulting tags |
|---|---|
| prores + `prores_metadata` bsf, no `write_colr` | bt2020 / smpte2084 / bt2020nc — **correct** |
| prores + bsf + `write_colr` | unknown / unknown / bt2020nc |
| prores, no bsf | unknown / unknown / bt2020nc |
| dnxhr, no `write_colr` | unknown / unknown / **bt709** |
| dnxhr + `write_colr` | unknown / unknown / bt2020nc |

ProRes needs the bitstream filter and must **not** get `+write_colr`, which downgrades the colr
atom to a legacy form that cannot express PQ — and my first fix, adding `write_colr` for DNxHR's
sake, silently broke the ProRes path that was already working. **DNxHR cannot be fully tagged
through this muxer at all.** Prefer `prores4444`. Untagged, Resolve reads a PQ HDR master as SDR:
a whole-image error that looks like a grading mistake.

Verified output: `prores 4444, yuv444p12le, bt2020nc / smpte2084 / bt2020`. The container is
12-bit and carries 10 bits of real data, because ffmpeg's *encoder* accepts only 10-bit input.

---

## 5b. Seeing it: the lightbulb stress test — `scripts/stress_proxy.py`

The artist asked to see the geometry working under hard, changing light, then specified the setups
he wanted. A visible 4 cm emissive bulb flies a tour of eleven named stations — front far, front
close, below chin, above head, camera left, camera right, behind rim, **key off with only a source
behind him** (twice), high three-quarter, front far — holding at each so it reads, with a full hue
sweep, a 120 BPM strobe and two hard blackouts throughout. 360 frames at 24 fps (the 150-frame
shot ping-pongs so the tour has time to play): `final/stress_proxy_clay.mp4`.

Three bugs the first render exposed, all worth recording:

1. **The fill panel was in shot.** Mesh emitters *are* visible to camera rays — the exact opposite
   of round 1's light-object trap — and a 60 cm white panel dominated the frame. Fixed with
   `visible_camera = False`.
2. **The "key off" stations rendered pure black.** With the bulb directly behind her it was
   occluded by her own body: physically correct and useless. Fixed by moving it well off-axis,
   raising backlight level (only a grazing sliver of silhouette returns any light), and adding a
   trace of ambient so she reads as a silhouette rather than an empty frame.
3. **The subject was too small** on the 28 mm demo lens; now 35 mm.

### Textured: her actual skin, hair, suit and microphone — `scripts/make_albedo.py`

The clay render proves the shadows are computed rather than photographed, but the artist wanted to
see real materials. That needs the plate's *own* lighting removed first, or it is lit twice:

    albedo = plate / E_source(neural normals)

`E_source` is the same assumed soft frontal key `relight_sh.py` uses — deliberately identical, so
proxy and gain map cannot disagree about the original light — and evaluating it against the neural
normals means brow, nose and cheeks each get the divisor their orientation earns. The proxy carries
UVs equal to each vertex's own image position, exact by construction because the shell *is* a
back-projected pixel grid, so the plate lands back on the geometry it came from at full plate
resolution rather than at vertex density.

A **red-dominance skin mask** gates subsurface and roughness, because a skin shader on a wool suit
or a foam windshield makes them glow like wax. A deliberate stand-in for
`facebook/sapiens2-seg-1b`, whose 29-class taxonomy is the real answer.

**This is a preview of the target architecture, not the composite path.** Every pixel is still a
plate pixel scaled by a renderer-computed value, so identity cannot drift — but it renders the
proxy's *geometry*, so it inherits the proxy's silhouette and softness. The real composite leaves
the plate's own pixels in place and only multiplies them by a ratio. Still round 3's job.

---

## 6. The artist's green-highlight question, answered with measurements

*"The last frame of the relit stress test looks like she has chroma-green highlights in her hair
and around her suit — or is that a virtual light?"*

**It is the virtual light**, and the hair is *also* residual spill. Both, and they compound.

| frame | plate (keyed+despilled) core green-excess | gain map core RGB | gain green-excess |
|---|---|---|---|
| 0 | −0.0225 | 1.69 / 1.03 / 0.51 (warm) | −0.067 |
| 75 | −0.0219 | 2.07 / 1.83 / 1.59 | −0.000 |
| 149 | −0.0232 | 0.41 / **0.53** / 0.46 | **+0.101** |

The plate's green content is flat across the shot, so nothing in the source turns green late; the
gain map does, left-weighted (+0.155 left vs +0.058 right), matching the visible wash. But on the
matte fringe the despill leaves a constant green tail (p99 +0.052 at both frames), and the
fraction of edge pixels that end up green-positive goes **13.5 % → 68.3 %** between frames 0 and
149. Leftover green screen, made visible by a green light. Two independent fixes: soften the
rig's hue accent, and tighten fringe despill.

---

## 7. New traps found

1. **`OPEN_EXR_MULTILAYER` is listed in the `file_format` enum but rejected** unless
   `image_settings.media_type` is set to `'MULTI_LAYER_IMAGE'` first. The error names the very
   tuple that contains the value.
2. **opencv cannot read multilayer EXR** — `cv2.imread` returns `None` with no error. Previews
   must be written from inside Blender.
3. **Transformers upgrades drag numpy with them.** `-U transformers` wanted numpy 2.5.3; pin it.
4. **Sapiens2 is not spherical-imagery-agnostic** in the way it matters here: it must see the
   *desqueezed* plate, or a face is 1.6× too narrow and the geometry is plausible and wrong.
5. `validate_normals_convention.py` originally fell back to a camera rotation of (90, 0, 0) that
   silently *agreed* with `build_hdri_scene.py`. A fallback that happens to be right is not
   evidence — it now reads a recorded rotation and reports which source it used.

---

## 8. Honest assessment against the bar

**Not there, and the gap is now specific rather than vague.**

What genuinely works: keying, depth, face tracking, photographic environment, per-frame
backgrounds, exact-CFR mastering, SH level/colour tracking, validated normals, proxy geometry
that casts correct self-shadows, and an anamorphic camera model that is exact in geometry and
measured in bokeh.

What does not: **nothing the proxy renders reaches the plate yet.** Constraints 1–5 in CLAUDE.md
are all 🟡 for the same reason — the ratio composite is unwritten. Until it is, the delivered
image is still gain-map relighting, which has a proven ceiling below the bar. Also open: MTF,
grain, chroma and motion-blur matching (9–12, all 🔴), per-material roughness, and temporal
stability (Sapiens2 measures 2.03°/frame against 1.85° for the depth gradient — slightly worse,
and uncontrolled in both).

---

## 9. Decisions carried forward — do not re-litigate

Everything in round 1 §5 still stands. Added this round:

- **The hard-constraint table in `CLAUDE.md` is the definition of done.** The artist asked for it
  explicitly and meant it "all the way". Do not report a stage as working while a row it depends
  on is red, and do not present a partial improvement as a satisfied row.
- **The artist's own rig is `capture_format.json`.** It does **not** apply to `reporter_a`.
- **10-bit is ffmpeg's ceiling**; the EXR sequence is the high-fidelity Resolve handoff.
- Neural normals stay, for coherence, correct extinction and — mainly — proxy detail, not for the
  directional-correlation number, which they did not improve.

---

## 10. Open questions for the artist

Round 1's four still stand (chrome + grey ball, record the lens, measure interocular distance, a
take with real depth movement). New:

1. **Confirm 1.6× on the lens barrel** and the **camera body, codec and colour profile** (log or
   Rec.709). Recording mode fixes the vertical field of view, which `capture_format.json`
   currently derives rather than knows.
2. **What T-stop do you actually shoot at?** The DOF magnitude rule is physics, but the taking
   stop is not recorded, and real bokeh departs from the ideal model depending on where the iris
   sits. One defocused test take settles it.
3. **Is the delivered frame the full 2.40:1 open gate**, or do you crop in post? This decides
   whether the background render is 3840×1600 or wider.

---

## 11. Starter prompt for round 3

> Read `rounds/round2.md` first, then continue.
>
> The proxy stage renders and provably casts self-shadows (shadow travels 14.5 % of face height
> between an uplight and a toplight, `logs/proxy_shadows.json`), but **nothing it renders reaches
> the plate**. That is the whole job this round:
>
> 1. **Write the ratio composite: `out = plate × R_d × S + G`.** Two renders of the proxy — lit by
>    the real room, and lit by the estimated `E_source` rig — give `R_d`. Include the **dedicated
>    subsurface passes explicitly**; a diffuse-only ratio silently drops all SSS in Blender 5.2.1.
>    `S` is the Shadow Catcher pass, `G` the additive specular delta. The plate's pixels must only
>    ever be scaled by renderer-computed values, so identity drift stays structurally impossible.
> 2. **Re-run the stress test through that path** and compare against this round's numbers: actor
>    luma swing 136.4 %, corr actor↔room luminance **+0.295**, warmth +0.295, control 0.86 %. The
>    honest test is whether a hard key now produces a *terminator* on the face rather than a broad
>    ramp — measure the gain's spatial gradient across the face, not just its left/right balance.
> 3. **Then constraint 2**: put something between a light and the actor in the environment render
>    and confirm the shadow lands on her and stops at her silhouette.
>
> Also worth doing, in rough order of value: per-material roughness and a real skin mask from
> `facebook/sapiens2-seg-1b`, replacing the red-dominance heuristic in `make_albedo.py`
> (constraint 5, and its 29-class taxonomy splits lips/teeth/tongue for free); temporal smoothing
> of the normals and the proxy, since neither is temporally controlled; and the plate-matching
> group (9–12) — measured MTF blur, 4:2:0 chroma match, measured grain before quantisation, and
> 180° motion blur on the CG.
>
> **Before the artist's own footage arrives, two things must exist that do not yet.** Both are new
> red rows from his capture spec, and both fail silently rather than erroring:
>
> - **`scripts/vlog.py`: V-Log → linear and V-Gamut → Rec.709, applied on ingest before keying.**
>   Verify the constants against Panasonic documentation and a shot ColorChecker rather than
>   inheriting them — `validate_sh_convention.py` is the pattern. Undecoded V-Log makes every
>   downstream stage wrong while still producing a picture.
> - **A 4:2:0 Long GOP ingest path**: chroma upsampled before keying, an all-intra intermediate so
>   the matte does not crawl on mosquito noise, and keying at native 6K with the downscale to
>   3840×1600 applied afterwards, so the 2.48× oversampling is spent on the matte.
>
> Do not spend the round improving the gain map. It has a measured ceiling and round 2 established
> where it is.
