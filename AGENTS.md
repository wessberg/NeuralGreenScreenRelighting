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
| 5 | **Specular response** — roughness, Fresnel, sheen on skin and fabric | A purely diffuse actor under a room full of practicals has no catchlights and no wet look. | 🟡 glossy passes rendering; no per-material roughness yet (needs `sapiens2-seg-1b`) |
| 6 | **Light direction, colour, level, falloff** | The baseline match. | 🟡 level and colour track; direction has a ceiling — see round 2 |
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

## House rules

- **Verify conventions, never assume them.** Axis, handedness and sign errors in this domain are
  invisible in a single frame and ungradeable afterwards. Test several directions, not one — a
  reflection error leaves the diagonal invariant and passes a single-direction check.
  `scripts/validate_sh_convention.py` is the pattern.
- **Measure before and after.** Every claim in `rounds/` is backed by a number from the repo.
  Prefer a control (relight off) over an unanchored figure.
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
| 4 | depth | `video_depth.py` | `relight/depth/*.npy` | Video Depth Anything; **affine-invariant disparity**, not metric |
| 5 | face track | `face_track.py` (facetrack venv) | `relight/face_track.json` | mediapipe; needs its own venv |
| 6 | **normals** | `neural_normals.py` | `relight/normals/*.npz` | see below |
| 7 | **albedo + skin mask** | `make_albedo.py` | `relight/albedo/`, `relight/skin_mask/` | plate ÷ `E_source(normals)`; needed before the plate can be re-lit, or it is lit twice |
| 7b | **proxy geometry** | `make_proxy_mesh.py` | `relight/proxy/*.npz` | see below; carries UVs |
| 8 | environment | `build_hdri_scene.py` (Blender) | `scene_library/<env>/` | anamorphic path is opt-in |
| 9 | light probes | `bake_light_probes.py` (Blender) | `<env>/lighting_sh.json` | SH-L2, Lambertian-convolved |
| 10 | relight | `relight_sh.py` | `relight/controls/*.npz` | gain map; **has a measured ceiling**, see round 2 |
| 11 | proxy render | `render_proxy.py` (Blender) | `relight/proxy_render/*.exr` | self-shadows, SSS, glossy passes |
| 11b | stress demo | `stress_proxy.py` (Blender) | `composite/stress_proxy*/` | lightbulb tour; `--textured` uses the albedo |
| 12 | composite | `composite_shot.py` | `composite/`, `intermediate/linear_4k/` | loads background **per frame** |
| 13 | master | `master_export.py` | `master/*.mov` | ProRes 4444 / DNxHR 444, 4:4:4 10-bit |
| 14 | delivery | `hdr_mux.py` → `validate_final.py` | `final/*.mp4` | HEVC 4:2:0 PQ + HDR10 SEI |

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

### Anamorphic (`scripts/anamorphic.py`)

All 1.6× arithmetic lives in one module; never retype the factor. Opt in per shot with
`camera.anamorphic` in `shot.json` — the stock reporter plate is spherical and must not inherit
it. Geometry is exact (3840×1600 desqueezed = the full 36×24 mm gate squeezed); bokeh direction
was measured, not assumed (`aperture_ratio` 1.6 → 1.6:1 vertical oval).
