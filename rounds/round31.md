# Round 31 handoff — released DiffusionRenderer runs locally; both generative eye trials rejected

**Date:** 2026-09-09. Starts from round 30. Production defaults unchanged.
**Delivered:** an isolated diffusion runtime, released SVD DiffusionRenderer
inverse/forward inference fitted into 16 GB, five G-buffers and two lighting arms
at 384 and 512, a narrowed reflection-spot trial, three review videos, exact
source-fallback verification, and screenshot-to-frame matching.
**Not delivered:** an acceptable eye repair. **Both trials were rejected by the
artist on sight.** No scene reflection replacement, no validated world
registration, no temporal constraint on the generated patch, no pupil/iris
material distinction and no production change. Photorealism is not met.

**Provenance of this file.** The round-31 working session terminated on a
provider usage limit at 13:56, before it wrote this handoff. Every number below
is read from the logs under
`music_videos/reporter_test/shots/reporter_a/logs/r31_*.json` and from the
checkout, not from recollection. Section 3.3 recovers the output of the last
command, which completed (exit 0, 5.07 s) but was never read by the agent that
launched it. No inference was re-run while writing this.

## 1. Artist steering and scope

The artist changed the standing constraint mid-round. Verbatim:

> Generative is OK for selective things, especially if we've burned like 20
> rounds not being able to fix something. And I've already said I like
> DiffusionRenderer. That said, if you think our current approach is getting
> close to a working solution, let's continue

Round 30's optical solver was judged not close, so it was dropped. This
**supersedes round 30's ban on generated performer texture** and the strict
pixel-provenance requirement. Selective generation is authorized for chosen
regions. Still required: recognizable identity, gaze, expression and timing;
original pixels preserved outside the repair region; exact source fallback for
failed repairs. Generic input support and no capture references are unchanged.

Both results were then rejected:

> The diffusion eyes look terrible and completely unrealistic?

> I mean it "kind of" removed the reflection, but instead now we have a light
> brown spot inside the pupil

This does **not** reinstate a blanket ban. It rejects these two outputs.

## 2. What was built

Released SVD DiffusionRenderer at `dependencies/DiffusionRenderer`, verified at
commit `8fcf0057ad3422139cd53281037025ff725d34e9`. Weights are pinned by
revision in `logs/r31_weights.json`: inverse `nexuslrf/diffusion_renderer-inverse-svd`
`54243c27` (8 files), forward `nexuslrf/diffusion_renderer-forward-svd`
`bea8a656` (15 files). The runtime is isolated at
`dependencies/diffusion-runtime/.venv` so it cannot disturb the pipeline's
opencv: torch 2.8.0+cu128, diffusers 0.32.0, transformers 4.47.1, accelerate
1.3.0, peft 0.7.0, huggingface-hub 0.27.1, numpy 2.4.2, opencv-python 4.13.0.92
(`logs/r31_runtime.json`).

**It fits in 16 GB.** The documented configuration wants above 22 GB. With CPU
offload, VAE encode chunk 2, VAE decode chunk 4 and UNet forward chunk 1, peak
CUDA allocation is 3.83 GiB (inverse) and 5.43 GiB (forward) at 384, and
4.42 / 6.94 GiB at 512 — well inside the 5080. Cost for 24 frames: 99.75 s total
at 384, 165.25 s at 512.

Input is a fixed landmark-derived crop `[813, 142, 1209, 538]`, 396×396 native,
pixel aspect 1.0, source frames **700–723**, 20 steps, seed 31, model fps
condition 7 (a conditioning value, not a review retime). Display-sRGB into the
model, decoded back to scene-linear Rec.709 for masked blending; 0 source
channels clipped. Masks are **unvalidated landmark proposals**, recorded as such.

Five inverse G-buffers per trial — basecolor, normal, depth, roughness, metallic —
then two forward arms: `neutral` (radiance 0.1) and `scene`
(`warm_restaurant_night_4k.exr`, sha `a15bfa2e…`, strength 1.0, 180° rotation).
The environment log records `world_registration_validated: false` and
`finite_scene_geometry_reflections: false`: this is direct latlong sampling, not
a registered scene. The SVD code and weights are research/evaluation licensed;
this round is an evaluation, not adoption for delivery. Cosmos is a different
release and dependency stack and was not run.

## 3. Measured findings

### 3.1 Whole-eye replacement — rejected

Trials `r31_diffusion` (384) and `r31_diffusion512` (512), 24 frames each.
3,741,995 native pixels per arm are unchanged; source frames 708 and 709 (full
closure) fall back to exact source. Mean display change inside the masks is
27.0795 (neutral) and 28.0863 (scene) at 384, and 30.2736 / 19.1160 at 512.
**That number is a magnitude, not a quality score**, and the log says so.

The artist's verdict is the measurement that matters: flat golden irises,
distorted pupil boundaries and lost fine texture. The cause is upstream of
compositing — **the inverse pass already invents different iris and pupil
appearance in the generated basecolor**, and the forward renderer carries it
into the composite. Local colour matching downstream cannot correct invented
content. Recorded in `logs/r31_artist_review.json` with the screenshot hash.

### 3.2 Reflection spots only — also rejected

`r31_reflection_spots` restricts the edit to the detected reflection spots and
takes fill from the same generated basecolor, scaled by a per-eye linear gain
fitted on a surrounding ring. It is **removal only; no scene reflection is
added** (`scene_replacement: false`).

The edit is genuinely tiny: **9 of 24 frames edited, 15 fallback, at most 95
edited native pixels per frame, 680 in total** across both eyes, with every
pixel outside the spots exact. Edited-pixel mean display drops 85.7931 → 40.8054.
The 15 untouched frames are **source 704–718 consecutively** — the entire blink
interval is simply not repaired — and the left eye additionally abstains at 719,
leaving 17 edited eye instances, so some reflections remain visible.

**The fill is refit every frame with no temporal constraint, and the spread is
large.** Across the 17 instances the ring-fitted linear gain runs R 0.4787–0.8896
(median 0.7158), G 0.3195–0.6178 (0.4367), B 0.2762–0.6071 (0.3784). At the two
frames the artist compared directly, the left eye's gain is
`[0.8288, 0.5231, 0.3834]` at source 701 and `[0.4787, 0.3195, 0.3663]` at
source 722: **1.73× more red gain, and R/B warmth of 2.162 versus 1.307**. The
701 patch is both brighter and much warmer — consistent with the light brown
spot he saw — while 722 is darker and nearly neutral. Mask size is not the
difference (92 vs 89 pixels). This is a measured association between an
unconstrained per-frame gain and the observed frame-dependence; the generated
basecolor also differs per frame, so the gain is not established as the sole
cause.

### 3.3 Which frames the screenshots were — the recovered result

The artist asked whether the better-looking frame meant the patch had been fixed
and re-rendered. `audit_r31_feedback.py` answers it, and the answer was never
read. From `logs/r31_feedback_matching.json`:

| screenshot | best match | correlation |
|---|---|---:|
| 13:50:51 (brown pupil spot) | local frame 1 = **source 701** | 0.99433 |
| 13:53:25 ("much better") | local frame 22 = **source 722** | 0.99787 |

`video_sha256` `4a99ef8d…` matches the delivered slow review file:
**`video_not_modified: true`**. Nothing was fixed or re-rendered between the two
screenshots — they are two frames of one unmodified result, and the difference is
frame-dependent, per 3.2. Matching is inference from image correlation, not frame
metadata, and the log records that.

**The consequence for round 32:** this is a temporal-stability failure on top of
a content failure. A patch that reads acceptably at 722 and wrong at 701 is not
one that got unlucky in a single frame; it is one with no temporal constraint at
all.

## 4. Verification and review

`verify_r31_diffusion.py` passes on all three trials: all source hashes
unchanged, all generated image hashes match their logs, all arrays finite,
exact source outside every mask, and the author environment direction reproduces
to 3.28e-07 max abs error. No production module was modified;
`production_enabled: false` and `artist_accepted: false` are recorded in each log.

- `final/r31_diffusion.mp4` — 24 frames, 59.94 fps, 0.4004 s, max frame mean
  encode error 1.7931 display levels.
- `final/r31_diffusion512.mp4` — same length, error 1.7684; `_slow` at 12 fps, 1.4200.
- `final/r31_reflection_spots.mp4` — error 1.8372; `_slow` at 12 fps, 1.4538.

These are display-sRGB CRF12 reviews, not masters or artist acceptance. Encode
errors average unchanged pixels and are not repair scores. Artifacts are in
`composite/r31_diffusion`, `composite/r31_diffusion512`,
`composite/r31_reflection_spots` and `logs/r31_*`; raw command logs are
`logs/r31-*.log`.

```powershell
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/setup_diffusion_renderer.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/diffusion-runtime/.venv/Scripts/python.exe -B scripts/diffusion_eye_trial.py --shot music_videos/reporter_test/shots/reporter_a --stage inverse
& dependencies/diffusion-runtime/.venv/Scripts/python.exe -B scripts/diffusion_eye_trial.py --shot music_videos/reporter_test/shots/reporter_a --stage forward
& dependencies/diffusion-runtime/.venv/Scripts/python.exe -B scripts/diffusion_eye_trial.py --shot music_videos/reporter_test/shots/reporter_a --stage review
& dependencies/diffusion-runtime/.venv/Scripts/python.exe -B scripts/diffusion_reflection_trial.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r31_reflection_spots/review --fps shot --out r31_reflection_spots.mp4
& dependencies/diffusion-runtime/.venv/Scripts/python.exe -B scripts/verify_r31_diffusion.py --shot music_videos/reporter_test/shots/reporter_a --name r31_reflection_spots --spots
& dependencies/diffusion-runtime/.venv/Scripts/python.exe -B scripts/audit_r31_feedback.py --shot music_videos/reporter_test/shots/reporter_a --screenshots '<png>' '<png>'
```

`--size 512` and `--name` select the larger trial at the prepare/inverse/forward
stages. Nothing from round 31 is committed yet; `AGENTS.md` carries the rejection
summary, and the r17 and r29 work sitting in the tree is unrelated to this round.

## 5. Starter prompt for round 32

> Stay on eyes. Read round31.md and docs/r31-eye-trial.md. Selective generation
> is authorized and DiffusionRenderer's released SVD inverse/forward now runs
> locally inside 16 GB (99.75 s for 24 frames at 384), so the remaining problem
> is not access to a generative model. Both round-31 trials were rejected: whole
> eyes become flat and golden with distorted pupils, and the same generated
> basecolor restricted to reflection spots leaves a light brown spot inside the
> pupil. **Do not reuse that albedo as clean pupil material and do not correct it
> with another tint, gain or mask adjustment** — the failure is in the inverse
> pass, which invents iris and pupil appearance before any compositing.
> The next candidate must actually complete missing content inside a mask,
> constrained by surrounding anatomy, and must distinguish pupil from iris rather
> than filling both with one material; do not paint pupils brown or black.
> Constrain the fill temporally: round 31's per-frame ring-fitted gain swings
> 1.73x in red between source 701 and 722 with R/B warmth 2.162 vs 1.307, and
> that is what the artist saw as one frame looking wrong and another acceptable.
> Repair the blink interval or state plainly that it abstains; source 704-718 and
> the left eye at 719 are currently untouched. Scene reflection replacement is
> still unbuilt and still requires validated world registration —
> `world_registration_validated` and `finite_scene_geometry_reflections` are both
> false, so latlong sampling alone cannot place a real room reflection. Keep
> identity, gaze, expression and blink timing; keep original pixels exact outside
> the repair region; keep exact keyed fallback. Judge the result visually and
> temporally in a source/result video before claiming anything — a smaller edited
> area is not evidence of a repair, and neither is a mean display change. If you
> evaluate Cosmos, it is a different release and dependency stack: render and
> inspect it before calling it a solution. Produce a temporal review and
> round32.md with measured findings and the remaining physical work.
