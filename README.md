# Music-video virtual production workspace

This project turns the supplied greenscreen plate into a static-camera Blender interior shot. **Work is in progress; no final master has been validated yet.** This document is updated during production so another agent can pick up the actual pipeline.

## Intended use: multiple music videos, multiple directed shots

The user records **their own performances from different camera angles**, then gives creative/scene direction for each clip. This workspace may contain **several entirely different music videos**, each with many shots and potentially several Blender environments. Do not treat every incoming clip as another take of the example or assume that all clips belong to the same song or scene.

Use a hierarchy of **workspace → music video → shot/take**, with explicit references to reusable environments. A music video owns its audio, lyrics, broad creative brief, shared music timeline, edit and delivery specifications. A shot owns its source recording, angle/camera match, selected source range, offset against the music timeline, shot-specific direction, matte, geometry/relighting caches, scene variant and renders. Environments may be shared across shots or wholly different, as the creative brief requires.

The target organization is described in `music_videos/README.md`; `music_video_brief.template.json` captures video-level direction. The current example's existing files remain at the project root so the in-progress render is undisturbed; `music_videos/proof_of_concept/project.json` explicitly catalogs that example. New videos/shots should use their own directories and must not overwrite this example.

## Experimental eye observations

Run `scripts/iris_observations.py --shot <dir> --review` after keying and face tracking.
It derives native eye crops, orientation and pixel aspect from that shot, and preserves
source eyes while its masks remain unvalidated. It does not yet remove or replace
reflections. See [eye-reflection architecture and neural prior art](docs/iris-prior-art.md)
and [round 14](rounds/round14.md) for current measured limits. Released EllSeg and
RITnet run in the isolated neural environment; setup is recorded in
[the neural evaluation](docs/neural-eye-evaluation.md). The experimental
`neural_iris_correspondence.py` samples original donor pixels and excludes fitting
and query interpolation footprints across its entire search. Its reporter audit
earns no reflection repairs; candidate masks and their abstention thresholds remain
unvalidated. The source-keyed fallback and production defaults remain unchanged.

## Running without per-frame AI credits

**All production processing here is local:** CorridorKey, Video Depth Anything, the photographic relighter, Blender/Cycles, audio feature analysis and FFmpeg/NVENC. These scripts contain no paid model API calls and need no API key. After dependencies and weights are installed, running the saved pipeline without a cloud agent consumes no cloud-model tokens; it still uses local compute, electricity and storage.

Codex's planning, creative interpretation, code changes, conversation and image inspection use the agent's plan allowance or API billing, even when the tools execute on this machine. [Official pricing/usage documentation](https://learn.chatgpt.com/docs/pricing). This production session therefore is not free of agent usage, although the GPU processing itself is local.

For lower recurring usage, create/revise the scene and shot settings once, then run repeat processing as ordinary batch jobs. `scripts/reproduce.ps1` orchestrates the saved example without invoking Codex or another language model. It can reuse the key and scene with `-SkipKey -SkipScene`; the keyer also skips completed frame caches. Future multi-shot development should generalize configuration and cache invalidation, add a local queue/UI, and reserve optional agent calls for creative changes or unusual failures. Natural-language scene directing remains a separate capability; a completely local language model is a possible future component and is not installed or promised equivalent performance here.

### Audio, lyrics and music-driven lighting

Audio and lyrics are useful project inputs. Preserve an untouched song/master file, its sample rate, any stems, lyric text and any supplied timecoded lyrics. They can support:

- **Performance/edit synchronization:** align each angle's camera audio, clap or other reference to a common music timeline; store the exact offset, any measured drift and alignment confidence. Keep photographed singing/playing synchronized instead of silently time-stretching it.
- **Lighting cues:** derive candidate transients/onsets, beats, downbeats, intensity envelopes and sections from the song. Translate approved artistic choices into deterministic Blender light-energy/color animations, emissive practicals or scene changes.
- **Creative and editorial direction:** use verse/chorus structure, lyric meaning, repeated phrases and timecoded lines to suggest shot changes, environments, lighting transitions and visual themes. Lyrics without timestamps provide meaning, not reliable frame-accurate timing; align them to audio before using them as exact cues.
- **Cross-shot consistency:** evaluate every lighting cue in **music-video time**, then convert into each shot's local frame time using its offset and exact rational frame rate. Changing camera angle must not reset a shared beat/light sequence.

Automatic beat/onset/section estimates are suggestions with uncertainty, especially for free-time singing or ambiguous rhythm. Creative direction determines which cues are used and how strongly. Prefer smooth musical envelopes and deliberate accents over rapid full-frame flashing. Source shutter/motion blur and fractional frame rates must be accounted for when rendering animated lights.

The relighting controls must be regenerated from the **same animated Blender lights at each shot frame**. Applying music-driven light changes only to the background would break actor integration. The current proof of concept uses steady lighting and source audio; it does **not** claim to implement a synchronized multi-shot edit or animated audio-reactive relighting yet. Those are explicit next development requirements, not user-operated manual pipeline steps.

An initial reusable feature extractor is available in `scripts/analyze_music.py`. It runs locally through FFmpeg and NumPy, producing 10 ms RMS/bass/mid/high envelopes and candidate spectral onsets with timestamps. The example analysis is `music_videos/proof_of_concept/analysis/audio_features.json`. It is **not a verified beat grid**, lyric alignment, or an artistic lighting animation. Run it on a future supplied master with `--audio <path> --output <video-analysis-path>`; the agent should then map selected musical features to the creative brief and shared Blender/relighter cue timeline.

## Project direction: future footage and creative briefs

**This is intended to become a reusable virtual-production project, not a pipeline limited to the supplied example.** It will receive other greenscreen recordings, potentially with different frame rates (including fractional rates), shutter speeds/angles and motion blur, white balance/color temperature, resolutions, camera heights, focal lengths, viewing angles and actor positions. Each clip may also have its own creative direction: desired environment, scene construction, mood, reference imagery, lighting, composition and grading.

Future agents should begin each shot by inspecting its own source and creative brief. The values in this proof of concept — 30 fps, 10 seconds, green-background hint thresholds, estimated 52 mm lens, level camera and night listening room — are **shot-specific choices**, not global defaults that can safely be reused without inspection.

- **Timing:** preserve each clip's exact rational frame rate and frame timing. Derive frame counts and audio duration from the selected range; do not assume 30 fps or 300 frames. Handle variable-rate input deliberately and record any conversion.
- **Shutter/motion blur:** read shutter metadata where available or record a supplied/estimated value and its uncertainty. Keep captured motion blur in the matte; tune edge treatment and temporal filtering to the shot. Changing frame rate does not recreate a different shutter speed.
- **White balance/color:** discover transfer, gamut, bit depth and range independently. Respect log/HDR sources instead of treating every plate as sRGB. White balance metadata is often absent; record assumptions or supplied reference information and match the environment without destroying skin or wardrobe colors.
- **Camera/geometry:** re-estimate lens, horizon, height, orientation, scale, floor and proxy placement for each angle. A locked camera at a different angle is still a new camera match. A moving camera would require an explicitly revised scope; this example does not implement tracking.
- **Creative direction:** the supplied brief should drive scene modeling, materials, light layout, framing and grade. The listening room is only this proof of concept's design. Keep the photographic actor and temporal stability as hard constraints while choosing a suitable relighting strength.
- **Isolation:** preserve original footage; keep each new shot's source, settings, scene, caches, outputs and validation separate. Log source hash, tool/model versions, exact parameters and any creative revisions. Do not overwrite this completed example when a new clip arrives.

`shot.json` records the active example. `shot_brief.template.json` records the intended per-shot input/creative contract. **The present production scripts are still tuned for this single excerpt.** The template is documentation for future generalization, not a claim that arbitrary clips already run correctly without camera, color, keying and scene adaptation. Those adaptations are the agent's responsibility, not a request for manual user configuration.

## Workspace layout

```text
music_videos/<video>/                 project.json, audio, lyrics, analysis, edit, final
  shots/<shot>/                       shot.json + source key relight intermediate composite final logs
scene_library/<environment>/          .blend, lighting_sh.json, probes/, background/, renders
assets/hdri/                          downloaded CC0 panoramas + provenance json
footage/                              authoritative original recordings, never modified
templates/                            brief templates for new videos and shots
scripts/                              the pipeline; every stage takes --shot
logs/                                 project-wide diagnostics only (per-shot logs live in the shot)
dependencies/                         third-party checkouts and model weights
```

**Every stage is addressed by `--shot`.** There is no default; running a script without it lists
the available shots. An environment is named by the shot (`environment.dir` in `shot.json`), so
several shots can share one, and a shot can be re-lit by pointing it at a different one.

```powershell
$SHOT = "music_videos/reporter_test/shots/reporter_a"
$PY   = "dependencies/CorridorKey/.venv/Scripts/python.exe"

& $PY scripts/calibrate_key.py --shot $SHOT --write     # derive screen thresholds from the plate
& $PY scripts/key_plate.py     --shot $SHOT --test      # then without --test for the full range
& $PY scripts/video_depth.py   --shot $SHOT
& dependencies/facetrack/.venv/Scripts/python.exe scripts/face_track.py --shot $SHOT

# Environment: photographic HDRI rather than a modelled room
& 'C:\Program Files\Blender Foundation\Blender 5.2lender.exe' -b --factory-startup `
    --python scripts/build_hdri_scene.py -- --shot $SHOT
& 'C:\Program Files\Blender Foundation\Blender 5.2lender.exe' -b <env>.blend `
    --python scripts/animate_lights.py -- --shot $SHOT
& 'C:\Program Files\Blender Foundation\Blender 5.2lender.exe' -b <env>.blend `
    --python scripts/bake_light_probes.py -- --shot $SHOT --frames 150
& 'C:\Program Files\Blender Foundation\Blender 5.2lender.exe' -b <env>.blend `
    --python scripts/render_background.py -- --shot $SHOT --frames 150

& $PY scripts/sh_project.py      --shot $SHOT           # probes -> SH, with axis validation
& $PY scripts/relight_sh.py      --shot $SHOT
& $PY scripts/composite_shot.py  --shot $SHOT           # per-frame background, writes preview video
```

### Traps found the hard way, all now guarded in code

- **Blender light objects are invisible to camera rays.** `visible_camera` defaults to `False`
  and forcing it `True` still renders nothing, so a panoramic light probe cannot see them.
  Probe-visible emitters must be **emissive meshes**. Symptom: every probe byte-identical.
- **The equirect axis convention is not the obvious one.** `x = sin*sin, y = -sin*cos` maps
  +X to -Y: a reflection across `y = -x`, which leaves the XY diagonal invariant, so a diagonal
  spot check passes while every axis is 90 degrees out. Run `scripts/validate_sh_convention.py`.
- **`render.fps` is an int16**, so `60000/1001` is silently clamped to `32767/1001`. Use
  `fps = 60, fps_base = 1.001`.
- **`use_persistent_data` serves stale lighting** when stepping frames in one process.
- **Blender 5.x removed `action.fcurves`** (slotted actions: layers -> strips -> channelbag)
  and `scene.node_tree`, and `image_settings.media_type` must be set before `file_format`.
- **CorridorKey's `despill_strength` does not affect `result['fg']`**, only `result['processed']`.
- **Keying thresholds do not transfer between plates.** Derive them with `calibrate_key.py`.
- **The 8-bit plate is 4:2:0 and upscaled 2x**, so it carries ~1920 px of real detail across a
  4K frame. Native CG is about twice as sharp, which alone reads as composited.

## Source and editorial decision

- Authoritative supplied file: `MongolianThroatSingingmeme(GreenScreen).mp4`. Preserved untouched and copied to `source/plate.mp4`.
- Source: 1920×1080, H.264 Main, 8-bit 4:2:0, 30 fps, 866 frames / 28.866667 seconds, stereo AAC at 48 kHz. Full metadata: `logs/source_ffprobe.json`.
- The footage contains a seated musician, green clothing, and a string instrument against a bright artificial green backing. Existing photographic softness and cutout edges are already baked into the source.
- The later source contains substantial reframing/digital zoom. The proof of concept uses **frames 0–299: the first 10 seconds at the original 30 fps**, retaining the original audio and photographic actor framing. This is a deliberate short, locked-off excerpt, not the full source duration.
- No focal-length or camera-sensor metadata was present. Camera estimate: 52 mm on a 36 mm sensor, height 1.02 m, actor distance 4.3 m, level horizon, no camera animation. Feet are outside the original crop; a hidden seated proxy supplies approximate floor/stool occlusion.

## Discovered machine and software

- Windows, Ryzen 9 9950X3D, 64 GB RAM; RTX 5080, 16 GB VRAM; NVIDIA driver 616.64.
- Existing Blender **5.2.1 LTS**: `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`.
- Existing Git 2.53.0 and `uv` reused. Global Python 3.14 is left alone.
- No existing CorridorKey or relighter was found in the inspected application/project locations. **Updated discovery:** the user installed the official Blender Lab MCP 1.0.0 extension during production. It is at `%APPDATA%\Blender Foundation\Blender\5.2\extensions\lab_blender_org\mcp`. Initial tests used Blender background Python; subsequent scene control uses the installed MCP add-on's real TCP bridge.
- Official CorridorKey cloned into `dependencies/CorridorKey`. Its isolated `.venv` uses Python 3.13.14, PyTorch 2.8.0+cu128, and torchvision 0.23.0+cu128. A real CUDA matrix operation verified RTX 5080 support.
- Existing imageio FFmpeg 7.1 was used during discovery. A portable FFmpeg **9.0.1 essentials** build was installed inside `dependencies/ffmpeg`; it includes ffprobe and working HEVC NVENC support.
- Repositories and model weights are local; the source footage is never uploaded.

## Pipeline and design decisions

1. **Decode** the excerpt into 16-bit RGB PNGs, retaining higher-precision YUV-to-RGB conversion. The source itself remains 8-bit; this does not recover lost source detail.
2. **CorridorKey v1.0**, official Hiera/GreenFormer + CNN refiner, CUDA, native 2048×2048 inference. The script calls the actual documented `CorridorKeyEngine.process_frame` API. Brightness/chroma hints distinguish the saturated backing from the actor's green wardrobe. Separate half-float EXR alpha and straight foreground are retained. Original source RGB replaces the inferred color in the opaque core; only edge pixels use unmixing and restrained despill. Global green despill is disabled to protect the costume.
3. **Original Blender listening room** with oak boards, plaster, acoustic slats, recessed city window, sofa, credenza, ceramics and linen practical. Cycles uses OptiX on the RTX 5080. Static clean/shadow backgrounds are rendered once, avoiding animated render noise. Hidden seated proxy geometry is invisible to the camera and glossy rays, used only for interaction.
4. **Video Depth Anything Small** supplies temporally aligned geometric guidance. Its real temporal model uses 32-frame windows, 10-frame overlap, scale/shift alignment and an 8-frame blend. The DINOv2 attention calculation is adapted at runtime to equivalent PyTorch CUDA scaled-dot-product attention, avoiding the old xformers/PyTorch dependency pins. No model weights are altered.
5. **Custom 2.5D irradiance-ratio relighting**, driven programmatically by Blender light positions, RGB, power and camera data in `blender/lighting.json`. Intended to apply restrained low-frequency diffuse lighting, warm/cool response and soft edge/back illumination to the original photographic pixels. Geometric inference does not generate image texture. Further details will be recorded after representative tests.
6. **Scene-linear composition and HDR grading**, then 3840×2160, HEVC Main10, Rec.2020/PQ, 1000-nit target. Encoding and quantitative validation are pending.

### Relighter evaluation

The official [LiveLight](https://github.com/mayuelala/LiveLight) repository was cloned and its actual stage-3 interface, configuration and dependencies inspected. It provides released temporal diffusion code and weights, but its default 512×512 output and PyTorch 2.1/xformers pins make it a less direct fit for this already-soft, detailed photographic plate and Blackwell GPU. LiveLight **is not used for the output**, and no claim is made that it was benchmarked locally. Likewise, Relit-LiVE's published recommendation is at least 24 GB VRAM. The chosen approach uses [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) for temporal geometry and a deterministic photographic relighter, prioritizing exact face, hands, fabric, instrument and timing retention.

## Geometry stages added in round 2

Full detail in `AGENTS.md` and `rounds/round2.md`. The short version: a gain map multiplies the
plate, so it can never produce a shadow. These stages put real geometry in front of a ray tracer.

```powershell
$P = "dependencies/CorridorKey/.venv/Scripts/python.exe"
$B = "C:\Program Files\Blender Foundation\Blender 5.2lender.exe"
$S = "music_videos/reporter_test/shots/reporter_a"

# Surface normals (Sapiens2). The convention validator MUST run first and pass.
& $P scripts/validate_normals_convention.py --shot $S
& $P scripts/neural_normals.py --shot $S

# Proxy geometry: matte silhouette + metric depth + normal-integrated fine relief
& $P scripts/make_proxy_mesh.py --shot $S

# Ray-trace it, and prove the shadow moves when the light does
& $B -b --factory-startup --python scripts/render_proxy.py -- --shot $S --frame 0 --test uplight
& $P scripts/validate_proxy_shadows.py --shot $S

# Lightbulb tour: 11 named stations, hue sweep, 120 BPM strobe, blackouts
& $B -b --factory-startup --python scripts/stress_proxy.py -- --shot $S

# Anamorphic camera model (the artist's SIRUI Saturn 35mm, 1.6x)
& $P scripts/anamorphic.py                                    # print the derived model
& $B -b --factory-startup --python scripts/anamorphic.py -- --validate

# Editorial masters for Resolve
& $P scripts/master_export.py --shot $S --report
& $P scripts/master_export.py --shot $S --codec prores4444
```

Two things in here are load-bearing and easy to get silently wrong:

- **`logs/normals_convention.json` is not optional.** Camera-space normals map to Blender world
  as `world = (c0, -c2, c1)` — an axis swap, not the sign flip that was predicted. Round 2's
  validator ranks all 48 signed permutations against a sphere, then corroborates with flip
  equivariance and anatomy. `neural_normals.py` refuses to run without it.
- **Monocular depth over-estimates facial relief by ~2.2x.** Unanchored, the proxy came out
  68 cm deep. The relief scale is anchored on the face at 0.085 m of visible-surface depth.

## Current files and commands

Run from this project root in PowerShell. Paths are intentionally local to the project except the existing Blender executable.

```powershell
# Dependency setup, already completed
uv sync --extra cuda --project dependencies/CorridorKey

# Exact extraction used (frames 0–299)
& dependencies/ffmpeg/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe -hide_banner -y -i source/plate.mp4 -frames:v 300 -pix_fmt rgb48be -start_number 0 source/frames/%06d.png

# Representative key frames, then resumable full excerpt
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/key_plate.py --test
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/key_plate.py

# Temporal depth test (44 frames) or full excerpt (300)
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/video_depth.py --test
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/video_depth.py

# Blender scene test: 960×540, 48 samples. Final: 4K, 384 samples.
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' -b --factory-startup --python scripts/build_scene.py -- --test
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' -b --factory-startup --python scripts/build_scene.py
```

### Blender MCP control (current preferred route)

The add-on's source was inspected to use its actual interface. Start its official background command (a project helper is already running locally on `127.0.0.1:9876`):

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' -b --online-mode blender/listening_room.blend --command blender_mcp --host 127.0.0.1 --port 9876
# From a second terminal:
python scripts/blender_mcp_client.py --inspect
python scripts/blender_mcp_client.py --script scripts/build_scene.py --test
python scripts/blender_mcp_client.py --script scripts/build_scene.py
```

`scripts/blender_mcp_client.py` implements the installed add-on's NUL-delimited JSON transport with the required `type=execute`, `code`, and `strict_json` fields. It restricts script paths to this project and saves request/response transcripts under `logs/mcp_*.json`. This talks to the MCP add-on directly; no separate Codex-native Blender tool is registered in this session. User-wide Blender preferences and Codex configuration are not changed. The direct background Python commands above remain a reproducible fallback.

Do not run the memory-intensive inference and Cycles stages simultaneously. The key script skips completed foreground EXRs when resumed. `--test` deliberately refreshes its representative frames. Original files in the project root are not overwritten.

## Directory map

| Directory | Contents |
|---|---|
| `source/` | Original copy and 16-bit extracted excerpt |
| `key/alpha`, `key/fg`, `key/qc` | Linear alpha, straight linear photographic foreground, inspection images |
| `blender/` | Original `.blend`, exported lighting, float clean and proxy-shadow backgrounds |
| `relight/` | Temporal depth and relighting controls |
| `composite/` | Composition tests, inspection images and previews |
| `intermediate/` | Source contact sheets and eventual high-bit-depth master |
| `final/` | Final viewing outputs; currently pending |
| `scripts/` | Reproduction and quality-control code |
| `logs/` | Metadata, inference, rendering, encoding and validation evidence |
| `dependencies/` | Portable repositories, weights, FFmpeg and isolated Python environment |

## Explicit color handling

The source has no transfer/primaries tags. It is treated as display-referred sRGB-transfer / Rec.709-primary footage, documented as an assumption. CorridorKey receives floating-point sRGB and linear hints. Its original-core-preserved RGB is converted exactly once through the inverse sRGB transfer into scene-linear Rec.709 and saved as straight half-float EXR. Alpha is always linear coverage. Blender EXRs are scene-linear Rec.709, without baking the AgX viewing transform. Compositing is intended in linear float, followed by a documented absolute-luminance HDR grade, Rec.709→Rec.2020 primary conversion, ST 2084 encoding and explicit full-RGB→limited-YUV conversion. HDR completion evidence will replace this provisional section when available.

## Running production log

- **Discovery complete:** found the sole MP4, inspected source contact sheet and decoded all 866 frames without errors. Confirmed source properties and existing later zoom.
- **Dependencies complete:** official CorridorKey and Video Depth Anything Small weights downloaded; CUDA verified. Official LiveLight source retained for evaluation provenance. Full FFmpeg/ffprobe installed locally.
- **Extraction complete:** 300 frames in `source/frames/`, RGB16 PNG.
- **First key test complete:** frames 0, 60, 150, 240 and 299 processed at 2048. Opaque-interior alpha means exceed 0.99999; no interior pixels below 0.9 except eight pixels in frame 60, to inspect visually. Approximately 1.2 seconds per frame after initial warmup. No full-sequence quality claim yet.
- **Scene script written:** next steps are visual key inspection, test scene rendering, temporal depth, relighting/compositing iterations, full-resolution export and final decode/metadata QC.
- **Key and geometry tests inspected:** key retains wardrobe, bow and instrument; original opaque pixels are preserved. Video Depth Anything processed 44 test frames in about 1.2 seconds of inference, with 1.87 GB peak CUDA allocation at the selected guidance resolution.
- **First scene test rendered:** native geometry and both shadow/clean EXRs produced successfully on OptiX. City/window details will be refined before final rendering. Use the project Python environment for EXR inspection: the global OpenCV 5 build did not read these EXRs, whereas project OpenCV 4.13 reads them correctly.
- **User steering — MCP:** installed Blender Lab MCP extension discovered, its actual protocol inspected, and a project-local background bridge started. Scene revisions and final rendering are being migrated to this bridge.
- **MCP verified:** `logs/mcp_inspect.json` confirms the live listening-room scene, Cycles and `bl_ext.lab_blender_org.mcp`. The scene was rebuilt and test-rendered through this bridge. Blender's background MCP context lacks the convenience `context.object` property; the script now uses `context.view_layer.objects.active`.
- **Scene refinement:** moved the simple city geometry much farther away, increased its variation and reduced distracting window reflections. Both test background EXRs were regenerated through MCP.
- **Full temporal depth complete:** 300 frames, 6.25 seconds inference, 1.87 GB peak CUDA allocation. CorridorKey full excerpt continues independently of CPU relighting development.
- **Relighting/composition implemented:** `scripts/relight_controls.py` exports smooth gain/specular maps using actual light positions, orientation, area size, power and RGB. It applies forward/backward optical-flow confidence only to lighting controls. `scripts/color_pipeline.py` and `scripts/composite.py` explicitly implement scene-linear composition, absolute-nit HDR grading, Rec.2020 conversion and PQ before Main10 NVENC. Representative composite inspection is in progress.
- **MCP iteration corrected:** visual inspection caught old geometry left by context-dependent selection/deletion when rebuilding via the background MCP bridge. Replaced it with direct project-scene datablock deletion. Rebuilt and verified 816 objects, no unassigned mesh materials and no camera-visible proxies. The overlapping white geometry is gone.
- **First photographic composites inspected:** source face, hands, bow and green wardrobe remain intact; cool illumination is visible on the left/front, with warm right-side influence. The low-resolution lighting tests are under `composite/`. Lampshade material is being improved with actual translucency before full-resolution rendering.
- **Future-shot requirements captured:** added the explicit project direction above, `shot.json` for this example, and `shot_brief.template.json` for future technical metadata and creative direction. Different shutter, white balance, frame rates and camera angles must be handled per shot.
- **Full key complete:** all 300 alpha/foreground EXR pairs are on disk. GPU inference completed in approximately 315 seconds. Full-sequence preview and numerical key/detail checks are running.
- **HDR path verified independently:** `scripts/test_hdr_encoding.py` encoded and decoded known PQ luminance patches through the same NVENC Main10/full-to-limited path used for final output. Recovered 1/10/50/100/200/500/1000-nit values are within 0.7%; black at 0.005 nits is within 0.0002 nits. Evidence: `logs/hdr_encoding_test.json`. This checks real pixel encoding, not merely color tags.
- **Sequential review complete:** inspected five 12-frame regions beginning at frames 0, 50, 140, 230 and 288, plus full-frame comparisons. No obvious generated-detail changes or lighting flicker were seen. Minimum core alpha mean: 0.9999847; largest detected internal low-alpha component: 4 source pixels. Maximum opaque source-detail MAE: 0.0000331 in linear RGB. Maximum frame-to-frame mean RGB gain step: 0.00312. Evidence: `logs/preview_qc.json` and `composite/sequence_*.jpg`.
- **4K Blender renders complete via MCP:** both clean and proxy-shadow backgrounds rendered at 3840×2160, 384 Cycles samples, OptiX and denoising. Inspected a full-resolution photographic composite. Scene maximum is about 5.64, retaining emissive/highlight headroom above diffuse white.
- **Final export running:** 300 4K scene-linear half-float EXRs are being retained while CUDA grades and converts to PQ/Rec.2020 and NVENC encodes HEVC Main10. Standard mastering-display and content-light-level SEI will be attached before MP4/audio muxing and full validation.
- **Music-video intent recorded:** video-level audio/lyrics/creative briefs, angle synchronization, shared musical lighting cues and separate video/shot ownership are explicit requirements. The current example remains steady-lit; no unsupported claim of a completed multi-shot music-video editor is made.

## Known source and method limitations

The source has baked softness, 4:2:0 compression, an artificial green replacement and cropped feet. A 4K output cannot restore native 4K facial or textile detail. Single-view video depth and a seated proxy approximate geometry and illumination; this is not a recovered 3D actor or a measured reflectance/light-stage reconstruction. Strong changes to captured hard shadows are deliberately avoided. These limits will be assessed again on the final shot.

## Final output

Pending rendering and validation. Do not interpret scripts, tests or the `.blend` as a completed delivery.
