# Current eye research and a general reflection strategy

Checked against author sources on 2026-09-09. This is a targeted update, not an
exhaustive ranking of all 2026 research. Tracker/segmenter failures in this repo
do not establish that contemporary neural reconstruction cannot recover an iris.
The artist explicitly includes drums and guitars: eye-only architecture is too narrow.

## What is available and what was actually run

| Method | Author evidence | Practical status here |
|---|---|---|
| UnReflectAnything, CVPR 2026 | [Paper](https://arxiv.org/abs/2512.09583), [code](https://github.com/alberto-rota/UnReflectAnything), [weights](https://huggingface.co/AlbeRota/UnReflectAnything/tree/main/weights). Frozen DINOv3 features, highlight prediction, token inpainting and diffuse decoding. | Full checkpoint downloaded; all 601 state entries loaded exactly. Trained highlight head ran on 100 native eye crops. A full forward pass also ran on one author-bundled sample. Performer diffuse reconstruction and instruments were not evaluated. |
| Reconstructing Realistic and Relightable Eyes, WACV 2026 | [Paper abstract](https://openaccess.thecvf.com/content/WACV2026/html/Khademi_Reconstructing_Realistic_and_Relightable_Eyes_WACV_2026_paper.html), [author page](https://wkhademi.github.io/). Hybrid explicit cornea / implicit eye-interior model with refracted camera and lighting rays. | The author describes light-stage captures. No code/checkpoint link was found on the checked author/paper pages. This is evidence of reconstruction capability under its capture setup, not a deployed ordinary-video model here. |
| Seeing the World through Your Eyes, CVPR 2024 | [Project](https://world-from-eyes.github.io/), [author implementation](https://github.com/HadiZayer/eyenerf). Joint corneal pose, reflected radiance field and iris texture optimization. | Unrun. Per-capture optimization, not a universal pretrained iris restorer. Code documents a static camera/moving subject and CUDA 11.8 environment, requiring compatibility work on this Blackwell GPU. The main reported goal is reflected-scene recovery. |
| EyeNeRF, SIGGRAPH 2022 | [Author project](https://ligethz.github.io/eyenerf_website/). Eye-region animation, novel views and relighting using multiple cameras/lights. | Unrun, capture-dependent. Distinct from HadiZayer's repository above despite similar naming. Neither paper licenses a claim that all ordinary performance footage is recoverable. |
| EyePreserve | [Author repository](https://github.com/CVRL/EyePreserve). Learned iris deformation with biometric identity objectives and available model downloads. | Unrun; documented inputs are near-infrared images and target masks. Dilation deformation/biometric identity is different evidence from visible RGB corneal-reflection separation. |

The initial local catalogue had treated UnReflectAnything's README warning as
decisive. It is contradicted by its actual author-hosted files and current loader.
Availability must be checked through checkpoint download and compatible inference,
not just examples or release prose. This correction changes the next experimental
priority; it does not establish this model as the best or as a production repair.

## UnReflectAnything deployment

- Author code: `363ebb5c87d9f4f49b63f744fded8d3786a051b3` (package 1.1.1).
- Full checkpoint HF revision: `ed889890c625698c8665d872fc02987bafd1b134`.
  SHA256: `12c0cf4518a39325b7bfe2bed04a96395189f2643c2f90b7b79360a659dc739b`.
- Encoder mirror revision: `f43714afb20addc93acad542bf6ea0376b6f7900`.
- Isolated runtime: `dependencies/unreflect-runtime/.venv/Scripts/python.exe`;
  torch 2.9.1+cu128, torchvision 0.24.1+cu128, numpy 2.4.2, OpenCV 4.13.0.
  Transformers is the author's pinned commit
  `2fe43376cdde02b7ffcf117e6eb9aa4375fb2dd1` (4.56.0.dev0).
  The complete installed versions are recorded in `scripts/data/r19-unreflect-runtime.txt`.
- The embedded training config sets `LOCAL_PRIOR_WEIGHT=0.2`, while the packaged
  YAML says 0.5. Use the embedded config. This forward parameter is not protected
  by strict state-dict shape checks. The author's current loader handles it.
- `unreflect_highlights.py` loads with `weights_only=True` and the explicit DotMap
  allowlist, strips checkpoint prefixes with the author's helper, and verifies all
  loaded tensors exactly. Local file paths replace online initialization paths.
- Encoder/preprocessing/highlight head are unchanged. 64x64 observed display RGB
  is processed at the author's 896px resolution; the resulting probability map
  is sampled back at native half-pixel coordinates. Enlargement adds no evidence.
- 100 inputs, plus two repeated source-977 checks: 28.986 seconds of inference,
  3,539,735,552 bytes peak CUDA allocation. 0/12 reflection and 0/30 iris probes
  exceed 0.3. Only three native pixels exceed it over all 100 crops. Catchlight
  probe probabilities range about 0.00073-0.04420. Do not lower the threshold
  after these results or call it a reconstruction test.

An initial integration check used the first bundled SCARED surgical frame. The
full network produces finite diffuse output and its highlight output matches our
head exactly; max highlight probability is 0.95416. No diffuse image was saved or
shown. This establishes that the head is not universally silent, not performance
on ordinary scenes. After the artist questioned those samples, ordinary scene or
object images became the preferred inputs for further checks. The verifier now
requires an explicit `--integration-image` for fresh full-model checks; by default
it preserves that historical result and verifies saved reporter maps only. No
surgical image appears in either round-19 review video.

## Proposed general-material architecture (not implemented)

1. Estimate original material/lighting contributions across the performer and
   carried objects. A general highlight predictor can propose affected regions;
   it cannot certify clean radiance or automatically authorize RGB replacement.
2. Preserve observed texture through temporal samples where correspondence is
   supported. Validate learned diffuse estimates separately against observations:
   a visually plausible inpainted feature is not proof of the original detail.
3. Supply geometry, pose, visibility and a material model to Cycles. Skin, wet
   eyes, lacquered wood, fabric, chrome hardware and cymbals need different
   responses. In particular, a diffuse-only plate cannot represent polished metal;
   the scene-dependent reflected environment must be reconstructed/rendered too.
4. Couple original-light removal and new-light rendering. Replacing a catchlight
   without removing its source, or removing a sheen without its new material
   response, repeats the rejected composite. Preserve exact source on uncertain
   performer regions under the artist's accepted fallback decision.
5. Evaluate source-versus-result temporal footage with material-specific failures:
   instrument edges/strings, brushed highlights, broad reflections, hair, lips,
   blink boundaries, rapid sticks/hand motion and occlusions. No instrument footage
   was tested this round, and no special on-set reference is required.

This is the reason to investigate general inverse rendering and separation now.
More eye-tracking benchmarks alone cannot answer the artist's broader question.
