# AGENTS.md — working guide for any coding agent on this project

Canonical, provider-agnostic. `CLAUDE.md` imports this file; Codex CLI and other agents read it
directly. Put project knowledge HERE, never in a provider-specific file, so a fresh session on
any model starts from the same state.

## Read the latest round first

**Before doing anything else in this repo, read the most recent file in `rounds/`.**

```powershell
Get-ChildItem rounds/round*.md | Sort-Object Name | Select-Object -Last 1
```

Each round is the handoff from the previous working session: what was delivered and verified,
what is genuinely broken, decisions the artist has already made (do not re-litigate them), and a
starter prompt for the round you are about to do. The latest round supersedes this file and the
README wherever they disagree.

At the end of a working session, write the next `rounds/roundN.md` in the same shape, ending
with a starter prompt for round N+1.

**Round 8 reflection findings:** the original catchlights, crown sheen and lip gloss are still
unresolved. Disabling added GGX changes lip pixels by only 0.00955 display levels on average;
it is not the lip fix. Simple centre/scale temporal iris recovery fails a withheld-pixel test
(median relative RGB error 12.47–14.50%, p95 50.10–64.59%). Unclipped reflections are unresolved
sums, not automatically recoverable iris texture. `audit_r8_reflections.py` is an isolated
reporter diagnostic, not a production repair or validated removal mask. See `rounds/round8.md`.

---

## What this project is

A virtual-production pipeline that composites green-screen footage of the artist into Blender
environments, to produce music videos. The stated acceptance bar is **full photorealism** —
"if it doesn't look 100% authentic and like I was literally there, I don't want it anyway" — so
do not present incremental gains as success. Say plainly when an approach has a ceiling below
that bar.

The artist shoots his own performances at **24 fps**, usually in bust/medium framing with no
feet or floor contact in shot.

### His capture rig — `capture_format.json`

**Panasonic LUMIX S5 IIX, 6K 3:2 open gate 5952×3968, HEVC 10-bit 4:2:0 Long GOP 200 Mbps,
V-Log / V-Gamut · SIRUI Saturn 35 mm Full-Frame Anamorphic, 1.6× squeeze · 180° shutter at
24 fps · WB 5600 K · delivering 3840×1600 (2.40:1).**
Full detail and consequences are in `capture_format.json`; read it before writing any stage that
touches camera geometry or colour.

**Two things will bite, and both are silent.**

1. **V-Log is not linear and V-Gamut is not Rec.709.** Every stage here assumes scene-linear
   Rec.709 — keyer thresholds, depth and normal estimators, the SH irradiance ratio, the PQ
   grade. Feed V-Log in raw and all of them are wrong while still producing a picture: lifted
   midtones, crushed saturation, and every number in `rounds/` quietly invalid. Decode on
   ingest, before keying, and never apply a creative display LUT first — it is not invertible
   and throws away the highlight range the HDR grade exists for.
2. **The plate is anamorphically squeezed (1.6×) and Blender renders spherically.** Depth and
   normal estimators must see the *desqueezed* plate, or they infer a face 1.6× too narrow —
   plausible-looking and wrong. Defocus is an oval, not a circle. `scripts/anamorphic.py` holds
   every factor of 1.6; never retype one.

**A third that will bite the keyer: the plate is 4:2:0 Long GOP.** Chroma is stored at half
resolution (2976×1984) and a green-screen key is a *chroma* operation, so upsample chroma
properly before keying or every matte edge inherits a blocky staircase no threshold will fix.
Long GOP puts mosquito noise around exactly the high-contrast edges that matter — the silhouette
and every hair — so transcode to an all-intra intermediate **before** keying, or the matte will
crawl. Key at native 6K and downscale afterwards: with 2.48× linear oversampling the matte comes
out effectively supersampled, and hair edges get real fractional alpha instead of a hard
decision. Downscaling first throws that away.

The numbers all agree with no crop, which is worth noting because it is not a coincidence to be
re-derived each time: 5952/3968 = 1.5000 exactly, desqueezed by 1.6 that is 2.400:1, and the
delivery frame is 3840×1600 = 2.40:1. Shoot open gate — a 16:9 mode desqueezes to 2.84:1 and
forces a crop.

These values describe **his** footage only. `reporter_test/reporter_a` is online stock footage
shot by someone else at 59.94 fps, Rec.709, on an unknown spherical lens; its camera block in
`shot.json` is the authority there, and it must not inherit the rig above.

---

## Hard constraints — what "photorealistic" actually requires

The bar is not a feeling, it is this list. **A shot is not done until every row is green.**
Do not report a stage as working while a row it depends on is still red, and do not present a
row's partial improvement as the row being satisfied. If an approach cannot reach a row, say so
and name the approach that can — that is more useful than a better number on the rows it can
already do.

Status is for `reporter_test/reporter_a`, the test bed, as of the end of round 2.

| # | Constraint | Why it is non-negotiable | Status |
|---|---|---|---|
| 1 | **Self-shadowing** — nose onto cheek, brow onto eye, jaw onto neck | The single strongest cue that a light is real, small and close. A gain map *cannot* do this at any quality: shading is `dot(n,l)`, shadowing is occlusion. | 🟡 proxy casts real shadows, validated (shadow travels 14.5 % of face height between uplight and toplight); **not yet in the composite path** |
| 2 | **Received shadows** from scene geometry crossing the actor | A window mullion or a passing shape that stops at the actor's edge is an instant giveaway. | 🟡 the proxy is in the scene and occludes; needs an environment render rather than the test lights |
| 3 | **Cast shadows onto the environment** | Only matters where the actor is near a surface; bust framing usually hides it. Shadow Catcher pass. | 🔴 |
| 4 | **Subsurface scattering** in skin | Skin is not opaque. Without it, ears, nostrils and lit edges read as painted plastic. Light transport through a volume; no per-pixel multiply can fake it. | 🟡 `RANDOM_WALK_SKIN` + Jensen 2001 radii on the proxy, dedicated SSS passes enabled; not composited |
| 5 | **Specular response** — roughness, Fresnel, sheen on skin and fabric | A purely diffuse actor under a room full of practicals has no catchlights and no wet look. **And the ratio composite cannot remove the plate's OWN speculars**: `plate × R` with a smooth R recolours a highlight but cannot move or delete it, so the original room's sheen survives every relight — the artist spotted this on sight. Needs specular/diffuse separation on the plate, not a setting. | 🟡 **unchanged, and a documented dead end behind it.** Round 5 built a dichromatic separation, claimed it worked, and was rejected on sight: it removed 26.9 % of tooth energy and 52.7 % of sclera energy, blocked up dark hair, and painted reflections over eyelids. Reverted; defaults are round 4's. The original room's sheen, lip gloss and catchlights are all still in the photograph — see the rejection addendum in `rounds/round6.md` and `rounds/round7.md` |
| 6 | **Light direction, colour, level, falloff** | The baseline match. | 🟡 ratio composite delivers it; `E_source` now MEASURED from the plate rather than guessed (24 % relative RMS correction) |
| 7 | **Anamorphic lens match** — 1.6× squeeze, oval bokeh, correct FOV | The plate is anamorphic and Blender renders spherically. A silent mismatch here is geometric, not aesthetic. | 🟢 `scripts/anamorphic.py`: geometry exact, bokeh measured at 1.6:1 vertical (error 0.0). Blur *magnitude* still needs matching against a real defocused take |
| 8 | **Depth of field** matched to the taking lens | Nobody shoots a dim interior deep-focus; a sharp background is the "computer-ey" tell. | 🟡 f/2.0 in the environment render |
| 9 | **MTF / sharpness match** | The CG background is ~2× sharper than the plate. Measured blur on the background, not the actor. | 🔴 |
| 10 | **Grain match**, generated before quantisation | Also dithers 8-bit banding. σ ≈ 0.0042 sRGB, generated at 1080p and upscaled 2× because the plate is. | 🔴 |
| 11 | **Chroma subsampling match** (4:2:0 behaviour on the plate) | Sharp CG chroma against subsampled plate chroma separates the two layers. His own footage is 4:2:0 Long GOP, so this also governs the keyer: upsample chroma and de-Long-GOP before keying. | 🔴 |
| 12 | **Motion blur match** — 180° shutter at 24 fps | CG rendered without matching blur strobes against a correctly blurred plate. | 🔴 |
| 13 | **Edge integration** — no green fringe, correct edge alpha, no dark halo | Round 2 measured 68 % of hair-fringe pixels going green-positive once a green light was applied. | 🟡 despill works; fringe compounds under coloured light |
| 14 | **Temporal stability** — no flicker, no crawling normals or matte | A per-frame estimator with no temporal model will crawl. Sapiens2 measures 2.03°/frame. | 🟡 measured, not yet controlled |
| 15 | **Identity preservation** — the actor's own pixels, never a generative reconstruction | Non-negotiable for the artist's own face. Structurally guaranteed by ratio compositing; broken by any diffusion relighter in the output path. | 🟢 architecture enforces it |
| 16 | **Perspective and scale match** — lens, distance, height | Wrong focal length reads as a wrong-sized person. | 🟡 ASSUMED for the reporter plate, must be recorded on set |
| 17 | **Colour pipeline** — Rec.2020 / ST2084, 1000-nit master, ProRes 4444 / DNxHR 444 for Resolve | | 🟢 validated end to end, including that the ProRes encoder silently drops `-color_trc` and needs the `prores_metadata` bitstream filter |
| 18 | **V-Log / V-Gamut decode on ingest** | His camera records log over a wide gamut. Undecoded, every downstream stage is wrong in a way that still looks like a picture. | 🔴 not built; constants not yet verified against Panasonic documentation |

Legend: 🟢 done and measured · 🟡 partial, with a known ceiling · 🔴 not built.

**Rows 1, 2, 3, 4 and 5 all require the same thing** — real geometry in a real renderer. That is
why the Cycles proxy stage is the architecture and not an enhancement.

---

## Orientation

```text
rounds/                               session handoffs — READ THE LATEST FIRST
music_videos/<video>/shots/<shot>/     shot.json + source key relight intermediate composite final logs
scene_library/<environment>/           .blend, probes/, background/, lighting_sh.json
footage/                               authoritative originals — never modify
assets/hdri/                           CC0 panoramas + provenance json
templates/  scripts/  logs/  dependencies/
```

**Every pipeline stage takes `--shot`.** There is no default; running a script without it lists
the available shots. A shot names its environment via `environment.dir` in its `shot.json`.

Two Python environments, deliberately separate:

| purpose | interpreter |
|---|---|
| pipeline (torch 2.8+cu128, opencv 4.13, numpy 2.4) | `dependencies/CorridorKey/.venv/Scripts/python.exe` |
| face landmarks (mediapipe 1.x needs opencv-contrib 5.x) | `dependencies/facetrack/.venv/Scripts/python.exe` |

Isolate any dependency that would disturb the pipeline's opencv — every EXR read depends on it.
Use `uv pip install --python <venv>/Scripts/python.exe <pkg>`; there is no `pip` in these venvs.

Blender is **5.2.1 LTS** at `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`.
A Blender Lab MCP add-on bridge may be listening on `127.0.0.1:9876`; `scripts/blender_mcp_client.py`
speaks it for read-only queries. Scene *building* runs under `blender -b --python`.

Hardware: RTX 5080, **16 GB, Blackwell sm_120** — requires CUDA ≥ 12.8 / torch ≥ 2.7. Any repo
pinning older torch will not produce a working kernel; it fails at runtime, not import.

---

## Blender 5.2.1 traps, verified in this repo

Each of these produces a wrong result with no error message.

1. **`scene.use_nodes = True` does NOT create the compositor node tree.** `scene.node_tree` is
   gone; `scene.compositing_node_group` stays `None` until you create one yourself:
   `bpy.data.node_groups.new(name, 'CompositorNodeTree')` and assign it.
2. **`CompositorNodeMixRGB`, `CompositorNodeMix` and `CompositorNodeMath` do not exist.** The
   compositor node set was restructured and there are no Mix or Math compositor nodes at all.
3. **Render Layers pass sockets are named in full words** -- `Diffuse Direct`, not `DiffDir`.
4. **Cycles has NO subsurface passes, and round 1's note to the contrary is wrong.**
   `use_pass_subsurface_direct/indirect/color` exist on the ViewLayer and accept `True`, but the
   Render Layers node exposes only Diffuse and Glossy. **Subsurface is folded into the diffuse
   passes**, so a diffuse-based ratio carries subsurface response rather than dropping it. The
   inherited warning said the opposite and would have sent a round chasing a pass that is not
   there.
5. **`OPEN_EXR_MULTILAYER` is rejected unless `image_settings.media_type` is
   `'MULTI_LAYER_IMAGE'` first**, and the error names the very tuple that contains the value.
6. **opencv cannot read multilayer EXR** -- `cv2.imread` returns `None` with no error, so any
   preview of a multilayer render must be written from inside Blender.
7. **Mesh emitters ARE visible to camera rays** -- the exact opposite of light objects, which are
   invisible even with `visible_camera = True`. Both halves of this bit: a light object threw
   shadows while staying invisible, and later an emissive bulb rendered a 266x187 disc of solid
   emission straight into an irradiance pass, which the composite then multiplied into the
   plate. **Anything an irradiance pass can SEE ends up scaling the photograph.**
8. **`use_persistent_data = True` serves stale results**, and the error GROWS through a sequence:
   rendering four frames with geometry and lights changing, persistent vs not, diverged by max
   7.55, 12.49, 14.83, 14.81 with mean error reaching 1.335. Confirmed, not inherited.
9. **`render.fps` is an int16**, so 60000/1001 silently clamps. Use `fps = 60, fps_base = 1.001`.
10. **`cv2.imwrite` to EXR rejects a float16 ARRAY.** Pass **float32** and let
    `IMWRITE_EXR_TYPE_HALF` do the conversion. Handing it a `np.float16` array warns
    *"Unsupported depth image for selected encoder is fallbacked to CV_8U"* and then asserts on
    `CV_8U` -- which reads like a codec problem rather than a dtype one. It cost a full 150-frame
    run, and the run reported **exit code 0** because the traceback was downstream of a pipe:
    check the tail of a background log, not just its status.
11. **numpy 2.x removed `ndarray.ptp()`.** `np.ptp(a)` is fine; `a.ptp()` raises `AttributeError`.
    The pipeline venv is numpy 2.4.

## GPU: what is accelerated, and why the card still looks idle

Measured on the RTX 5080, 2026-09-08. The artist noticed the GPU sitting at 11% and 62 W of a
400 W limit during a Cycles render and asked whether anything was actually accelerated.

**Already on the GPU:** Cycles through OPTIX, and every neural stage through torch (Sapiens2
normals in fp16, Video Depth Anything, CorridorKey). **Not on the GPU:** all the numpy/OpenCV
per-pixel maths, and video encoding. Note `cv2.cuda` reports **0 devices** in the installed
wheel, so no OpenCV call can be accelerated -- `scripts/gpu_image.py` uses torch instead, with a
CPU fallback that reports which device it actually used.

**But the idle card is not an acceleration problem, and this is the part that matters.** Cycles
render time barely depends on sample count:

| samples | persistent off | persistent on |
|---|---|---|
| 4 | 2.21 s | 1.89 s |
| 128 | 4.06 s | 2.90 s |

**About 2 seconds of every frame is fixed Blender overhead** -- scene sync, device upload, film
init, EXR write -- and only a fraction is ray tracing. A GPU that finishes its work in 0.3 s and
then waits 2 s reports low utilisation honestly. Confirmed from the other side: GPU 3.29 s vs CPU
5.01 s on the real proxy scene is only 1.5x, because both pay the same fixed cost.

Consequences, which are the opposite of the intuition:

* **Samples are nearly free. Spend them.** 128 samples costs 1.3 s more per frame than 4.
  Rendering noisy output to "save time" saves almost nothing.
* **Moving more work onto the GPU will not speed up rendering.** The GPU is not the constraint.
* **Fewer, larger renders beat many small ones.** Per-frame overhead is paid per render call.
* **`use_persistent_data = True` is NOT the way out.** It would save ~20%, and round 1 recorded
  that it serves stale lighting. That is now measured rather than inherited: rendering the same
  4-frame sequence with geometry and lights changing, persistent vs not, diverges and the error
  GROWS through the sequence -- max abs difference 7.55, 12.49, 14.83, 14.81 and mean error
  0.062, 0.186, 0.157, **1.335**. Do not enable it.

## House rules

- **Verify conventions, never assume them.** Axis, handedness and sign errors in this domain are
  invisible in a single frame and ungradeable afterwards. Test several directions, not one — a
  reflection error leaves the diagonal invariant and passes a single-direction check.
  `scripts/validate_sh_convention.py` is the pattern.
- **Measure before and after.** Every claim in `rounds/` is backed by a number from the repo.
  Prefer a control (relight off) over an unanchored figure.
- **No on-set reference is ever a requirement.** No chrome ball, no grey card, no colour chart
  as a precondition. The artist will not do it, and this is meant to run on other people's
  footage where no reference exists. If something needs measuring, infer it from the plate:
  `estimate_source_light.py` is the pattern -- it replaced the pipeline's last hand-written
  constant by fitting SH lighting to skin radiance against the neural normals.
- **State assumptions as assumptions.** The reporter camera match and `E_source` are both
  assumed and labelled so in `shot.json`. Do not let an assumption harden into a fact.
- **Never recalibrate the proof-of-concept shot's keying thresholds.** They are hand-set because
  that plate has a green costume, and they produced the keys behind a validated master.
- Commit with a message that explains *why*, including bugs found and what they would have cost.
- The README carries pipeline detail and the full list of Blender/library traps.

---

## The pipeline, stage by stage

Every stage takes `--shot`. Interpreter is the pipeline venv unless the row says Blender.
`P` = `dependencies/CorridorKey/.venv/Scripts/python.exe`.

| # | stage | script | writes | notes |
|---|---|---|---|---|
| 1 | ingest / extract frames | `shot_paths.py` resolves everything | `source/frames/` | rates are exact `Fraction`s |
| 2 | calibrate key | `calibrate_key.py` | `logs/key_calibration.json` | thresholds do **not** transfer between plates |
| 3 | key + despill | `key_plate.py` | `key/alpha/`, `key/fg/` | despill is the manual pass; CorridorKey's own `despill_strength` never touches `result['fg']` |
| 3b | ~~specular / diffuse split~~ | `separate_specular.py` | `key/specular/*.exr` | **EXPERIMENTAL, REJECTED — not in the delivery path.** Kept for diagnosis only; see below |
| 4 | depth | `video_depth.py` | `relight/depth/*.npy` | Video Depth Anything; **affine-invariant disparity**, not metric |
| 5 | face track | `face_track.py` (facetrack venv) | `relight/face_track.json`, `relight/face_landmarks.npz` + `.json` | mediapipe; needs its own venv. Also stores ALL 478 landmarks, blink, eyelid aperture, head pose |
| 5b | **corneal geometry** | `eye_geometry.py` | `relight/eye_geometry.json` | corneal sphere per eye, and the PLATE's key light solved from its own catchlights |
| 6 | **normals** | `neural_normals.py` | `relight/normals/*.npz` | see below |
| 7 | **albedo + skin mask** | `make_albedo.py` | `relight/albedo/`, `relight/skin_mask/` | plate ÷ `E_source(normals)`; needed before the plate can be re-lit, or it is lit twice |
| 7b | **proxy geometry** | `make_proxy_mesh.py` | `relight/proxy/*.npz` | see below; carries UVs |
| 8 | environment | `build_hdri_scene.py` (Blender) | `scene_library/<env>/` | anamorphic path is opt-in |
| 9 | light probes | `bake_light_probes.py` (Blender) | `<env>/lighting_sh.json` | SH-L2, Lambertian-convolved |
| 10 | relight | `relight_sh.py` | `relight/controls/*.npz` | gain map; **has a measured ceiling**, see round 2 |
| 11 | proxy render | `render_proxy.py` (Blender) | `relight/proxy_render/*.exr` | self-shadows, SSS, glossy passes |
| 11b | stress demo | `stress_proxy.py` (Blender) | `composite/stress_proxy*/` | lightbulb tour; `--textured` uses the albedo |
| 11c | **ratio irradiance** | `stress_proxy.py --ratio` (Blender) | `composite/ratio_render/*.exr` | neutral albedo at PLATE resolution and the PLATE lens |
| 11d | ~~catchlight~~ | `stress_proxy.py --catchlight` (Blender) | `composite/catchlight_render/*.exr` | renderer looks sound and its validator passes; **the compositing of it is rejected and `--catchlight` defaults to 0** |
| 12 | **ratio composite** | `composite_ratio.py` | `composite/ratio/` | `out = plate × R` by default. `--plate-specular remove` and `--catchlight` are rejected experiments |
| 12b | composite (gain map) | `composite_shot.py` | `composite/`, `intermediate/linear_4k/` | legacy path; loads background **per frame** |
| 12c | **review video** | `review_video.py` | `final/*.mp4` | what the artist watches; `--compare` puts the before on the LEFT with both halves labelled |
| 13 | master | `master_export.py` | `master/*.mov` | ProRes 4444 / DNxHR 444, 4:4:4 10-bit |
| 14 | delivery | `hdr_mux.py` → `validate_final.py` | `final/*.mp4` | HEVC 4:2:0 PQ + HDR10 SEI |

### Showing work to the artist (stage 12c)

**A round is not shown until there is a video.** Frames on disk are not a result he can judge —
the artefacts that matter here are temporal, and his eye is the acceptance bar. Rounds 1–4 each
re-derived the same ffmpeg line and re-probed the previous round's file for its settings; that is
what `review_video.py` exists to stop.

* Review encodes run at **24 fps regardless of the shot's rate**. The stress tour is 150 rendered
  frames, not 150 frames of a 59.94 fps performance. Pass `--fps shot` when judging real motion.
* They are **review only** — display-referred sRGB, `crf 12` because a normal crf hides the
  sub-pixel detail a round is usually arguing about. Delivery stays `master_export` → `hdr_mux`;
  never hand a review file to a grade.
* `--compare <dir>` puts the before on the LEFT and burns a label into each half. Do that by
  default when a round claims an improvement: a claim of "the dots are gone" is worth far less
  than two halves moving side by side.

### How normals are computed (stage 6)

`facebook/sapiens2-normal-1b` via `transformers` in the **pipeline venv** (needs ≥ 5.16; pin
numpy at 2.4.2 when upgrading, or it drags in 2.5.3 and disturbs opencv). ~0.42 s/frame on the
5080.

Three input requirements, none optional:
1. **Crop to the matte bbox, expanded to 3:4.** A raw 16:9 frame letterboxes into the model's
   1024×768 and leaves the face at ~150 px; the crop puts it near 600 px.
2. **Composite onto neutral mid-grey.** A saturated green backing is out of distribution and
   bleeds into the estimate exactly where hair detail lives.
3. **Convert linear → sRGB** before ImageNet normalisation. The keyed foreground is scene-linear.

**The camera→world convention is `world = (c₀, −c₂, c₁)`** — an axis swap, measured, recorded in
`logs/normals_convention.json`. `neural_normals.py` refuses to run without that file. Re-run
`validate_normals_convention.py` if the model, the checkpoint or the scene camera ever changes;
it ranks all 48 signed permutations against a sphere and corroborates with flip equivariance and
anatomy.

**What normals are for, and what they are not.** They give surface orientation, so they drive
`dot(n, l)` — *shading*. They cannot produce *shadows*: a cheek pixel's normal contains nothing
about the nose in front of it. Round 2 measured that swapping depth-gradient normals for neural
ones did **not** improve directional lighting. Their real value here is coherence, correct
extinction when lit from behind, and supplying the fine relief the proxy mesh needs.

### How proxy geometry is built (stage 7)

A displaced silhouette shell, in Blender world axes (+X right, +Y away from camera, +Z up, actor
at Y = 0):

* silhouette from the matte, displaced by metric depth using the **same** back-projection and
  calibration as the normals, so the two cannot disagree about where the actor is;
* closed with a flat back cap and stitched side walls, or it is invisible to rays from behind;
* **relief scale anchored on the face** at 0.085 m of visible-surface depth. Raw monocular depth
  over-estimates facial relief by ~2.2× and produced a 68 cm-deep bust;
* **fine relief integrated from the normals** (Poisson, DCT, Neumann). For `Y = f(X,Z)` the
  normal *is* the gradient field. Depth supplies scales above 5 cm, normals below — their error
  profiles are opposite. Without this the render is a mannequin with no nose ridge.

Validate with `validate_proxy_shadows.py`: it renders an uplight and a toplight and requires the
shadow to *move across the nose*, which is the one thing no per-pixel method can fake.

### The ratio composite, and why it is the delivery path (stage 12)

    out = plate × R,      R = E_target(rendered) / E_source(assumed)

**Never render the performer and show that.** A proxy has no pores, no individual hair strands,
no fabric weave, no corneal bulge. Rendering it produces exactly what the artist named on sight:
"a Vision Pro avatar", "a Sims character", "a plastic Barbie doll", "a 90s CGI game trailer". His
frame-by-frame verdicts mapped precisely onto how much of her the light revealed — heavily
shadowed frames read as acceptable, broadly lit ones did not — because a broad light exposes the
detail a smooth shell does not have.

The ratio composite renders the *irradiance field* on a neutral proxy, divides by the plate's own
assumed lighting, and multiplies the photograph. Every pore, hair and specular in the output is a
photographed pixel; the renderer contributes only the CHANGE in light, which is smooth almost
everywhere and sharp exactly where it should be — at a shadow edge.

Two rules that are easy to get wrong:

* **The ratio pass must use the plate's own lens and resolution**, or the multiplier does not
  line up with the photograph it multiplies. The wider demo lens in `stress_proxy.py` exists only
  to keep the bulb in frame and must not leak into `--ratio`.
* **Clamp the top of the ratio, not the bottom.** A shadow going dark is the point. `relight_sh.py`
  clamped both ends at 1.5 stops, capping the whole map at 2.83:1, which is exactly why it could
  never produce a terminator.

What it still inherits from the proxy: the *shape* of the shadows, so a nose shadow lands in
roughly the right place with the proxy's smoothness at its edge, and hair shadows as a mass. What
it does not inherit is the proxy's appearance, which is the whole difference.

### The specular layer (stage 3b) — BUILT, REJECTED, AND OFF. Read this before trying again.

`plate = D + S`. The ratio is a lighting scale and **only D obeys it** — a smooth `R` can recolour
a specular highlight but cannot move, reshape or delete one. So before round 5 every highlight in
the plate kept reporting where the green-screen room's lamps were, no matter where the Blender key
went. The artist reported this three times, in three places, without connecting them:

* *"I can kind of see the shine in her hair from the original green screen room."*
* *"the lips are somehow lit up more than the skin around them"*
* the corneal catchlight, still sitting at 11 o'clock on both irises — a plainly rectangular
  softbox, in a composite lit by a bulb.

**They are one bug** — and round 5's attempt to fix it made the picture much worse. The artist's
verdict on the result was *"It looks extremely terrible - all of it"*, and the measurements agree:
**26.9 % of tooth energy and 52.7 % of sclera energy removed**, 75.9 % of sclera pixels losing
over a quarter of their radiance, dark hair breaking into unstable patches. `--plate-specular`
defaults to `keep`. The stage is preserved for diagnosis and must not be re-enabled because the
files exist.

**The diagnosis that looked obvious was wrong, and that is the useful part.** It is tempting to
blame the neutral guard (`--cos-max`, a 1.8° cone) for failing to protect near-neutral teeth and
sclera. Measured, their local base colours sit **41.4° and 28.6°** from the illuminant — nowhere
near the guard, and **0 %** of sampled tooth or sclera pixels were inside it. Widening the cone
fixes nothing. The real fault is that **a local morphological opening is not a valid
material-colour reference**: it mixes materials across its window, so `b̂` is wrong wherever a
window straddles a boundary — which is exactly the mouth, the eye and every hair edge. Likewise
the hair blocks are not literal kernel stamps (measured patches are 5×4–7×4 px on a 15 px
kernel), though removal does swing 11.5 / 16.7 / 9.4 % across kernel widths 7 / 15 / 31.

If you try again, the bar is: **prove teeth, sclera and hair detail survive, and that the result
is temporally coherent, BEFORE enabling it.** `scripts/audit_r5_defects.py` is that test.

The parts below are what the attempt got right and are worth keeping:

* **The illuminant is measured, not assumed.** From the performer's own corneal glints — the eye
  is the plate's own chrome ball — cross-checked against the brightest near-neutral pixels. On
  the reporter plate the two agree to 0.3% while sharing no pixels and no mechanism.
* **The material colour is LOCAL**, from an alpha-aware opening, so there is no skin mask, no hair
  mask, and no assumption about skin tone, hair colour or wardrobe. Each pixel is fitted as
  `I = a·b̂ + s·Ĝ` with `a, s ≥ 0`: a rise in red alone (blood, lipstick) can only be `a`.
* **Skin gets *more saturated as it darkens*** — subsurface reddening — which mimics the
  dichromatic specular signature exactly. Measured over 58 819 cheek pixels of one material,
  `min/sum` runs 0.060 (p2) to 0.216 (p95). That is a genuine obstacle for any dichromatic
  method here: **a single unpolarised image cannot fully separate diffuse from specular on
  skin.** Cross-polarised capture can, and the artist will not shoot cross-polarised.

Near-neutral materials — a white shirt, grey foam, metal — are unseparable by *any* dichromatic
method. Round 5 reported that honestly for the wardrobe and then destroyed the teeth and sclera
anyway, because the failure there was not neutrality at all.

### Catchlights (stages 5b, 11d) — renderer sound, compositing REJECTED, `--catchlight` = 0

**The corneal geometry and the Blender render are validated** (`validate_catchlight.py`, direction and magnitude within 0.44–3.51 %). **The compositing of it is not.** The cap is a flat 2D disc centred on the iris with **no eyelid intersection**, so 5.8 % of its weight lands outside the eye opening and it painted silver arcs across her upper lids. Fixing the lid mask alone is not enough: source occlusion is still wrong (the proxy has no eye sockets) and it depends on the rejected removal stage to take the old reflection out.

**Do not paint a dot.** A catchlight is a mirror image of the source, so the honest way to make one
is to put a mirror where the cornea is and let Cycles find the source. Presence, size, shape,
colour and occlusion then all come from the scene: a bulb 0.5 m away gives a bright disc, a window
a soft rectangle, an overcast sky a broad wash, an unlit corner nothing at all.

**A distance threshold would get sunlight exactly backwards.** A mirror shows the source's
*radiance*, and radiance does not fall off with distance — so a distant lamp still makes a
catchlight, just a small one, and the sun makes a tiny brilliant spark rather than nothing. What
distance changes is solid angle, hence size.

**The cornea is a dielectric, not a mirror.** IOR 1.376, so Fresnel gives ~2.5 % at normal
incidence. A first attempt used a plain glossy BSDF with a white colour, i.e. 100 % reflectance,
and returned the whole cornea at 2.5–4.3 linear — it would have whited out both eyes.

**Keep or replace is computed, not set.** The plate's own catchlight is better than any synthetic
one when it fits — right shape, right softness, real lash occlusion. It fits only when the new key
is near the original. Both angles are measured: the plate's from its own catchlights
(`eye_geometry.py`, 54.6° off the camera axis on the reporter plate) and the scene's from the L1
moment of the rendered irradiance. Note the glint moves by **half** the angle the light moves, so
the tolerance looks generous and is not.

**Gaze estimation is resolution-dependent, and this is measured.** The iris ellipse's minor/major
ratio reads gaze directly, but needs a big iris: at 12.9 px radius it returned a confident, stable
**29.3° for a subject looking straight down the lens**, because one pixel of landmark asymmetry on
a 13 px circle is already 23°. It poisoned the light solve (40° scatter) before it was caught. So
the ellipse is used only above `--ellipse-min-radius` (the artist's macro close-up case) and head
pose plus iris displacement is the default below it. On the reporter plate that took the scatter
from 40° to 18.5°.

**The light POSITION does not come out**, and the gate for that is not the residual. Two corneas
63 mm apart cannot triangulate a source a metre away: the rays are near parallel and the
least-squares point collapses onto the corneas themselves *with a small residual*. The direction
is the usable output.

### Anamorphic (`scripts/anamorphic.py`)

All 1.6× arithmetic lives in one module; never retype the factor. Opt in per shot with
`camera.anamorphic` in `shot.json` — the stock reporter plate is spherical and must not inherit
it. Geometry is exact (3840×1600 desqueezed = the full 36×24 mm gate squeezed); bokeh direction
was measured, not assumed (`aperture_ratio` 1.6 → 1.6:1 vertical oval).
