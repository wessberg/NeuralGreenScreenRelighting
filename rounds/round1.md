# Round 1 — recovery from Codex, photographic environment, SH relighting

**Date:** 2026-09-08 · **Commits:** `fab8c56` … `92d67cb` (8) · **Working tree:** clean

This round began as a rescue. An OpenAI Codex session had exhausted its ChatGPT Plus limit
mid-turn after burning 6.17 M tokens, and the task was to read its transcript and carry on.
It ended with a restructured workspace, a photographic environment, and a working
spherical-harmonic relighter measured against an animated stress rig.

---

## 1. What was delivered, with evidence

### The proof-of-concept master was one command from done — and broken in four ways

Codex's final background job (`composite.py --final`) had **completed, exit 0 after 419 s**; it
ran out of credits before it could read its own result. The export then failed validation on
four separate defects, all now fixed:

| defect | effect | fix |
|---|---|---|
| B-frame POC reordering in a raw elementary stream | **the visible stutter** — 3 duplicate + 2 gapped presentation timestamps, 2 frames hidden behind an edit list | `-bf 0` |
| mov timescale 30000 | could not represent the composition offset; padded the track, dropped to 298 frames | timescale = frame-rate numerator |
| `-avoid_negative_ts make_zero` | 1-frame (33 ms) A/V desync once reordering was gone | removed |
| validator diffed timestamps in **decode** order | structurally could not detect duplicates — this is why Codex's own gate would have passed a stuttering file | check presentation order, assert the exact CFR tick set |

`music_videos/proof_of_concept/shots/night_listening_room/final/Night_Listening_Room_4K_HDR10.mp4`
validates **exit 0**: 3840×2160, 300 frames / exactly 10.000000 s, HEVC Main 10, Rec.2020 /
ST2084 PQ, MaxCLL 336, MaxFALL 22, full decode clean, 0.22 nit mean round-trip error.
Re-verified after the restructure.

### The pipeline is shot-agnostic

`scripts/shot_paths.py` resolves paths, frame rate, frame count, delivery spec and keying
thresholds from `shot.json` / `project.json`. Rates are exact `Fraction`s so 60000/1001 stays
exact. A delivery rate differing from source **errors** unless an explicit `conform` policy is
recorded — retiming is a creative decision, not a default. Delivery default is 24/1.

### Reporter test shot, fully prepped

150 frames (source 700–849 of 1200, chosen for the hand-to-ear gesture: self-occlusion and
moving hair stress temporal stability). Keyed 150/150, median `interior_below_0.9` = 15 px.
Depth 150/150 via Video Depth Anything Small (6.2 s, 1.87 GB VRAM). Face-tracked 150/150.

### Photographic environment replaces the modelled room

Poly Haven `warm_restaurant_night`, **CC0**, 24K available, 15 EV, verified unclipped (max
20785, 31.3 stops). Two copies split by Light Path `Is Camera Ray`: camera sees 16K, lighting
rays sample 4K. At 50 mm the hFOV is 39.6°, so the visible slice is **1802 px = 93.9 %** of the
plate's real ~1920 px (it is 1080p upscaled 2×). The usual objection to photographic backdrops —
softness — is inverted here, because the foreground is only 1080p. The camera is locked, so
there is no parallax to expose a flat backdrop.

**f/2.0 rather than f/5.6** does more visible work than anything else in this round. Nobody
shoots a dim interior at f/5.6, and the resulting defocus removes most of the "computer-ey" read
on its own.

### SH relighting, measured against a stress rig

Three SH-L2 probes per frame (head / chest / waist), baked in Cycles, blended by each pixel's
reconstructed world height. Every magic constant from the original is gone — `strength*0.13`,
`clip(0.35,1.5)`, and the `0.30 + 0.70*gain` damping that compressed all corrections by 30 %.
In their place: `gain = E_target/E_source × anchor`, then `tanh` in log2 space (monotone, unit
slope at zero, smooth saturation).

`scripts/animate_lights.py` drives every axis a relighter can get wrong: environment level ~50×,
key azimuth 200°, practical proximity 2.40 m → 0.45 m (~28× inverse-square), hue sweep,
beat-locked pulse at 120 BPM.

| measurement | value |
|---|---|
| actor luminance swing, relight **OFF** (control) | **0.86 %** |
| actor luminance swing, relight **ON** | **117.98 %** |
| room luminance swing | 233.79 % |
| actor↔room luminance correlation | −0.20 → **+0.39** |
| relighter gain, luma / warmth swing | 124 % / 150 % |

Videos: `music_videos/reporter_test/shots/reporter_a/final/stress_test_{compare,relit}.mp4`.

---

## 2. Traps found the hard way — read this section before touching Blender

Each of these produces **wrong results with no error message**.

1. **Blender light objects are invisible to camera rays.** `visible_camera` defaults to `False`,
   and forcing it `True` still renders nothing. A panoramic probe camera therefore cannot see
   them and captures only the world. Symptom: every probe byte-identical across an animated
   shot. Probe-visible emitters must be **emissive meshes** (still sampled as lights by Cycles).
2. **The equirect axis convention is not the obvious one.** `x = sinθ·sinφ, y = −sinθ·cosφ` maps
   +X→−Y and +Y→−X: a reflection across `y = −x`. That reflection **leaves the XY diagonal
   invariant**, so a diagonal spot-check passes while every axis is 90° out. Correct pairing is
   `x = sinθ·cosφ, y = −sinθ·sinφ`. Guarded by `scripts/validate_sh_convention.py` — six known
   directions in a black world, worst error 0.01°. *Never* validate this with one direction.
3. **`render.fps` is an int16.** `60000/1001` is silently clamped to `32767/1001`. Use
   `fps = 60, fps_base = 1.001`.
4. **`use_persistent_data = True` serves stale lighting** when stepping frames in one process.
5. **Blender 5.x removals:** `action.fcurves` (now slotted actions: layers → strips →
   `channelbag(slot)`), `scene.node_tree` (→ `scene.compositing_node_group`),
   `CompositorNodeComposite`; `image_settings.media_type` must be set **before** `file_format`;
   `glare_type` is now an input socket.
6. **CorridorKey's `despill_strength` does not affect `result['fg']`** — documented as "Raw
   Prediction (Straight)". Only `result['processed']` is despilled, and that is linear,
   premultiplied and garbage-matted, so this pipeline cannot use it without losing the
   original-core blend. All despill here is the manual pass in `key_plate.py`.
7. **Keying thresholds do not transfer between plates.** The proof-of-concept backing is
   *clipped* (1.000 / 0.929); the reporter's correctly exposed backing is 0.859 / **0.467** and
   missed a 0.48 threshold by 0.013 — classifying 0.26 % of the frame as backing while still
   reporting a healthy-looking interior alpha of 0.9955. Use `scripts/calibrate_key.py`.
8. **`composite.py` loads the background once** and reuses it for every frame, freezing the room
   behind a moving performer. `composite_shot.py` loads per frame. The legacy script is still
   frozen — fix or retire it.
9. **`blender/background_clean.exr` already existed and was unused.** With `background_shadow`
   it forms a shadow-catcher pair whose ratio is the actor's cast shadow. Half of a requirement
   was already rendered and thrown away. Now superseded by Cycles' `use_pass_shadow_catcher`.

Two of my own, caught by self-review before running: the probe height-blend accumulated rather
than interpolating (double-counting the lowest probe), and the depth normals had both tangent
terms negated — for `Y = f(X,Z)` the normal is `(fX, −1, fZ)`, and the inverted form reads
convex geometry as concave.

---

## 3. The architecture, and why

**Recommended and partially built: proxy geometry in Cycles, applied to the plate as ratio
passes.**

```
out = plate × R_d × S + G
```

`R_d` is a diffuse (+subsurface) ratio between two renders — a proxy lit by the real room ÷ the
same proxy lit by the estimated plate rig — `S` is the Shadow Catcher pass, `G` an additive
specular delta. **The plate's pixels are only ever scaled by renderer-computed values.** No
generative model sits in the output path, so identity drift is not "low risk", it is
structurally impossible. And a director's note ("move the practical 30 cm left, warmer") is a
Blender edit that propagates correctly through shadows, speculars and falloff; with IC-Light or
LiveLight it is a prompt re-roll.

Requirement coverage: only this architecture can produce shadows the actor **receives** (a
window mullion crossing the face). IC-Light might hallucinate something plausible but
uncontrollable; LiveLight, conditioned on a single point light, has no concept of an occluder.

**LiveLight is a no-go** — not mainly for its dead `torch==2.1` pin, but because it is a 512×512
model whose lighting conditioning is built from *one point light* and cannot express a room.

Independent corroboration: *"Yesnt: Are Diffusion Relighting Models Ready for Capture Stage
Compositing?"* (arXiv 2510.23494) studies exactly green-screen capture-stage compositing,
concludes **no** for pure diffusion decomposition, and endorses this architecture.

**Where generative models belong:** as *estimators* only — normals, albedo, roughness, material
class. Their pixels never reach the output, so they cannot drift identity.

---

## 4. Honest assessment

**Working:** level tracking, colour/hue tracking, keying, depth, face tracking, photographic
environment, per-frame backgrounds, the exact-CFR master path.

**Not working yet:**

- **Directional modelling.** The key sweeps 200° without visibly moving across her face. Gain
  shifts level and colour but does not sculpt. Depth-gradient normals are structurally flat;
  the fix is neural normals, not tuning. **This is the single biggest gap.**
- **She reads green late in the shot.** The relighter is *correctly* following a green accent
  under a nearly dark environment, but it looks like a cast rather than light. The rig
  over-weights that accent; also there is no competing key to anchor skin tone.
- **No shadows, specular or subsurface.** Needs the Cycles proxy stage.
- **MTF mismatch.** The CG background is ~2× sharper than the plate. Fix is a measured blur
  (σ ≈ 0.7–0.9 px at 4K on the background only), plus 4:2:0 chroma matching and measured grain
  (σ ≈ 0.0042 sRGB, generated at 1080p and upscaled 2× because the plate is).
- **`E_source` is assumed**, not measured — a soft frontal studio key. This is the weakest link
  in every ratio method.

Verdict against the stated bar ("100 % authentic or I don't want it"): **not there.** But the
remaining failures are specific and addressable rather than vague.

---

## 5. Decisions already made — do not re-litigate

- **Compare all four relighting approaches** (deterministic fix / neural intrinsics / IC-Light /
  LiveLight port) rather than picking one blind. Option 5 (proxy + Cycles) emerged as a fifth
  and is the recommended delivery path.
- **Non-commercial licences are acceptable** while experimenting — "use the best tool for the
  job". Note the top pick in each category was already permissive (Sapiens2, Marigold,
  Careaga/Aksoy). NC mainly unlocks **irradiance estimation** (V-RGBX / RGB↔X), which is the
  only route to `E_source` on footage already shot without a reference ball.
- **Target 24 fps**, configurable. The artist **shoots 24 natively**, so his own material
  conforms exactly. The 59.94 reporter clip is a test bed and stays at native rate.
- **Most shots will have no feet or floor contact** — bust/medium framing. Grounding effort
  belongs in lighting match, DOF, edge integration and scale, not contact shadows.
- **Scene visual quality can iterate later**; grounding the keyed footage comes first.
- Up to **two Opus 5 subagents** may be used.
- Do **not** recalibrate the proof-of-concept's keying thresholds. They are hand-set because
  that plate has a green costume, and they produced the keys behind the validated master.

---

## 6. Open questions for the artist

1. **`E_source`.** Shoot a **chrome ball and an 18 % grey ball on a slate at the head of every
   take**, and write down the light plot. This converts the weakest link in the chain from an
   inference problem into a measurement, costs ~30 s per take, and is entirely in his control.
2. **Record the lens.** The reporter camera match is an *assumption*: apparent facial scale
   fixes only the focal/distance ratio, so 35 mm/0.91 m, 50 mm/1.30 m and 85 mm/2.20 m all
   reproduce the measured 129.7 px IPD. 50 mm was chosen because it maximises panorama detail.
3. **Measure his own interocular distance with a ruler**, once. It removes the 63 mm adult prior
   (±5.5 % distance error) permanently.
4. **A take with real movement in depth.** The reporter clip's yaw-corrected facial scale varies
   only 3.59 %, so the true distance-driven irradiance swing is **7.15 %** — a weak test of
   falloff, though a good one for shadows/specular/SSS.

---

## 7. Next steps, in order

1. **Neural normals** — `facebook/sapiens2-normal-1b` (human-specialised, ~130 ms/frame,
   commercially licensed, installs on the existing torch 2.8 via `transformers ≥ 5.16`).
   **Crop to 3:4 around the matte bbox first**: a raw 1080p frame letterboxes to 768×432 and
   leaves the face at ~150 px. This is described in the research as the single highest-leverage
   decision in the pipeline. Composite onto neutral mid-grey before inference — a saturated
   green background is out of distribution and bleeds into the estimates.
   **Validate handedness on a rendered sphere before committing frames** — neural normals are
   camera-space, `(x, −y, −z)` for Blender, and that convention is inherited and never written
   down in the source repos.
2. **Material classes** — `facebook/sapiens2-seg-1b`, same install and forward cost. Its
   29-class taxonomy splits lips/teeth/tongue, giving wet-surface and enamel shaders for free.
3. **SSS by literature** — no released model estimates subsurface parameters. Use Jensen et al.
   2001 skin values with Blender 5.2's `RANDOM_WALK_SKIN`.
4. **Proxy shell + two-render ratio** — matte-silhouette shell (the silhouette is the most
   visually important property of a shadow, and the key already produces an accurate one),
   displaced by metric depth, closed with a back shell so it is not transparent to shadow rays.
   Then both Shadow Catcher directions. **Note: Blender 5.2.1 has dedicated
   `use_pass_subsurface_direct/indirect/color` — SSS is *not* folded into the diffuse passes, so
   a diffuse-only ratio silently drops all subsurface response.**
5. **Plate matching** — measured MTF blur, 4:2:0 chroma match, measured grain before
   quantisation (it also dithers the 8-bit banding).
6. **Retire or fix `composite.py`/`relight_controls.py`** — superseded by `composite_shot.py`
   and `relight_sh.py`, but still referenced by the proof-of-concept's reproduction path.

---

## 8. Starter prompt for round 2

> Read `rounds/round1.md` first, then continue.
>
> The gap to close this round is **directional modelling**: the SH relighter tracks level and
> colour well (118 % actor luminance swing against a 0.86 % control) but the key sweeping 200°
> does not visibly move across the actor's face, because depth-gradient normals are structurally
> flat.
>
> Build the neural-normal stage:
> 1. Crop to 3:4 around the matte bounding box and composite onto neutral mid-grey before
>    inference — both are required, not optional.
> 2. Run `facebook/sapiens2-normal-1b` via `transformers` in an isolated venv if it conflicts
>    with the pipeline's opencv 4.13, as `dependencies/facetrack/.venv` already does for
>    mediapipe.
> 3. **Validate handedness on a rendered sphere before processing 150 frames.** Camera-space
>    normals convert as `(x, −y, −z)` then rotate by the camera world matrix, but that convention
>    is inherited and undocumented upstream. `scripts/validate_sh_convention.py` is the pattern
>    to copy: test several directions, never just one — a reflection error leaves the diagonal
>    invariant and passes a single-direction check.
> 4. Feed the neural normals into `scripts/relight_sh.py` in place of `normals_from_depth`, and
>    re-run the stress test. **The number to beat is the actor↔room luminance correlation of
>    +0.39 and the actor↔room warmth correlation of +0.25**, and the qualitative test is whether
>    the key visibly sweeps across her face in `stress_test_compare.mp4`.
>
> Then start the Cycles proxy stage (matte-silhouette shell + back shell, two-render ratio, both
> Shadow Catcher directions), remembering that Blender 5.2.1 has dedicated subsurface passes that
> must be included in `R_d` explicitly.
>
> Also worth doing: soften the stress rig's hue accent, which currently turns the actor green
> late in the shot in a way that reads as a colour cast rather than as light.
