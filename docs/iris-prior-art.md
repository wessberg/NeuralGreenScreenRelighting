# Eye reflection relighting: reusable components and prior art

**Round 19 correction (2026-09-09):** the previous UnReflectAnything availability
assessment below was incomplete. Despite the stale GitHub README warning, the
author's Hugging Face repository contains a full checkpoint. It is now downloaded,
strictly loaded and its highlight head evaluated locally on 100 reporter eye crops.
All 601 state entries match. None of the 12 sparse catchlight anchors exceeds the
author's 0.3 threshold in this crop setup; diffuse reconstruction on the performer
was not evaluated. This does not rule out the model's reconstruction capabilities
or other materials. CoTracker3 has also now run in rounds 16-19. See the current
[research and general-material strategy](r19-material-research.md) and
[round 19 handoff](../rounds/round19.md). The artist explicitly includes drums,
guitars and other reflective objects in the product's scope.

Checked against authors' repositories and papers on 2026-09-08. None of the neural
models below has been installed or evaluated in this project during round 12.
Repository documentation advertising weights is not a verified download or inference run.

**Round 13 update:** EllSeg and RITnet author checkpoints have now been downloaded
and run in an isolated CUDA-compatible environment. See
[the reproducible evaluation](neural-eye-evaluation.md). EllSeg supplies complete
ellipse proposals; RITnet supplies visible-part proposals with observed failures.
Neither is validated for reflection removal. Other neural models below remain unrun.

**Round 15 update:** pretrained torchvision RAFT large was evaluated as a coordinate
estimator, with original donor RGB sampled separately. Nearby-donor hidden-pixel
error improves on 30 reused iris labels, but source-reflection support remains zero
and round-trip consistency passes points during a blink. See
[round 15](../rounds/round15.md) and its
[frozen protocol](r15-evaluation-protocol.md). This does not validate visible material
tracking or clean reconstruction. CoTracker3 remains unrun.

The intended product is **tracking + source-reflection removal + scene-reflection
rendering**, on arbitrary prepared shots. The reporter is an evaluation clip, not
the specification. A failed reporter experiment must not become a claim that
temporal reconstruction is generally impossible.

## Components worth evaluating before more custom detector work

| Component | Available prior art | What it supplies | What remains to establish here |
|---|---|---|---|
| Eye tracking and segmentation | [EllSeg](https://github.com/RSKothari/EllSeg), [RITnet paper](https://arxiv.org/abs/1910.00694) | Learned pupil/iris localization; EllSeg links pretrained checkpoints and a video inference path. | Visible-light performance footage, small irises, blinks, lashes and oblique views. An inferred full ellipse is not a mask of **visible** iris material. |
| Dense correspondence | [torchvision RAFT large](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.optical_flow.raft_large.html) | Released flow weights; tested in round 15 using explicit C_T_SKHT_V2, 12 updates and native coordinate conversion. | Flow and low forward/backward error do not certify iris identity, visibility or absence of source reflection. |
| Temporal points and visibility | [CoTracker3 authors' repository](https://github.com/facebookresearch/co-tracker) | Released offline/online trackers return point trajectories and predicted visibility. | Not run here. Needs explicit occluded-eye negatives and same-iris correspondence validation; predicted visibility is not automatically material truth. |
| Temporal removal baseline | [ProPainter](https://github.com/sczhou/ProPainter) | Released weights for flow propagation and Transformer video inpainting from frames and masks. | Whether repaired iris detail agrees with independent observations of that same eye; temporal consistency alone does not establish this. Its complete inpainting output is not guaranteed to consist of photographed donor pixels. The repo declares noncommercial use, so do not silently adopt it as a commercial production dependency. |
| Highlight separation baseline | [TSHRNet](https://github.com/fu123456/TSHRNet) | ICCV 2023 three-stage RGB specular removal; README links pretrained models. | Eye-specific detail, pupil/sclera preservation and temporal behaviour; this is single-image estimation, not a tracked-eye relighting system. |
| Multi-view corneal/iris decomposition | [Seeing the World through Your Eyes](https://world-from-eyes.github.io/), [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Alzayer_Seeing_the_World_through_Your_Eyes_CVPR_2024_paper.pdf) | Joint optimization of corneal poses, reflected scene and a shared iris texture field; public code is linked by the authors. | The paper evaluates the reflected scene reconstruction. Its radial iris regularizer may alter texture; it does not validate exact photographic iris restoration for our use. Small motion is a documented failure case. |
| Iris deformation | [EyePreserve](https://github.com/CVRL/EyePreserve), [DeformIrisNet](https://arxiv.org/abs/2207.08980) | Learned nonlinear pupil-dilation deformation with biometric identity objectives. EyePreserve links model downloads. | EyePreserve's documented input is near-infrared iris imagery plus a target mask. This is adjacent deformation work, not an RGB catchlight-removal model. Biometric identity preservation is weaker than exact pixel provenance. |
| Recent general highlight removal | [UnReflectAnything](https://github.com/alberto-rota/UnReflectAnything), [released author weights](https://huggingface.co/AlbeRota/UnReflectAnything/tree/main/weights) | General RGB highlight detection and learned diffuse reconstruction. Round 19 verifies strict checkpoint loading and executes its highlight head. | The README's unreleased-weights warning conflicts with the actual release. The reporter crop test does not validate masks or clean reconstruction; diffuse inpainting is not observed-pixel provenance. Instruments remain untested. |

The practical conclusion is to benchmark released components first, especially neural
eye segmentation/tracking. The sources establish relevant prior art, not a turnkey
system demonstrated to remove and replace corneal reflections on all performance
footage while preserving observed iris detail. We should neither declare the task
solved nor dismiss available neural methods without testing them.

## Shared pipeline and gates

1. Ingest into the shot's declared working space; preserve native sampling and the
   pixel-aspect mapping. The artist's camera constants are not generic defaults.
2. Track each eye and segment pupil, iris and occluders. Neural inference is allowed
   to estimate geometry, masks and correspondence without generating the performer's
   final appearance. Full inferred boundaries and observed visibility are distinct.
3. Match material points across observations of the **same eye**, accounting for
   pose and pupil changes. Validate on hidden native footprints. Neural tracking is
   a candidate replacement for fragile handcrafted correspondence, not an automatic
   confidence certificate.
4. Remove the original reflection using validated same-eye evidence. Keep the exact
   keyed source when detection, correspondence or coverage fails. The current sampler
   excludes the pupil, so a pupil-crossing repair needs a validated extension.
5. Render the replacement using the tracked cornea and actual scene lights in Cycles.
   Source shape, radiance, corneal orientation and Fresnel govern the reflection.
   Eyelids, lashes, brows and incident-light occlusion must be included. Do not paint
   a tracked white dot or use distance alone as a visibility rule.
6. Couple removal/replacement weights and fallback per eye over time. A failed eye
   keeps the original reflection and receives no added reflection. Validate the
   transitions in a source-versus-result video.

Existing `eye_geometry.py`, `stress_proxy.py --catchlight`, and `validate_catchlight.py`
provide corneal/rendering work already in the repo. Their rendering validation does
not validate the previously rejected compositor's flat corneal cap and missing lid/
source occlusion. `--catchlight` must remain off until those dependencies are fixed.

## What round 12's reusable code actually provides

`iris_observations.py --shot <dir> --review` consumes keyed EXRs and the tracked
landmark sidecar, derives per-eye crop sizes and orientation, and writes observation
NPZs plus a manifest. It uses the sidecar's pixel aspect to evaluate physical circles
in native coordinates without resampling the fitting input. It deliberately does not
read the reporter cache, its source-frame numbers, masks, colour statistics or labels.

`native_iris.py` is an experimental boundary fitter. `iris_correspondence.py` tests
torsion on disjoint real-image context and calls the existing `temporal_iris.py`
sampler. Numerical candidates remain unvalidated. The observation stage therefore
keeps source throughout; it is **not a finished automatic remover or scene-reflection
compositor**. Its array API and per-eye observations provide a replaceable boundary
for a pretrained detector, so testing one need not rewrite sampling or fallback.

The renderer and tracking must remain part of the plan even when an off-the-shelf
model is used for removal. The next useful comparison is a neural detector against
the recorded native failure cases, followed by correspondence/removal evidence and
occlusion-correct rendering—not further tuning the custom fitter to this reporter.
