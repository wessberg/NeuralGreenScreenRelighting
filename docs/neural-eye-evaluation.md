# Released neural eye models: round 13 evaluation

Both author checkpoints were downloaded and executed on the RTX 5080. This is
geometry/segmentation evaluation, **not reflection removal or replacement**.

The follow-on [round 14 correspondence evaluation](../rounds/round14.md) uses fresh
source-only sparse labels and a frozen donor/target split. It earns zero reliable
real repairs and exposes a nearly closed eye passing the experimental candidate
gate. Its reusable sampler retains the distinct geometry/visibility semantics and
exact source fallback described here.

| Model | Author source / pinned commit | Checkpoint SHA-256 |
|---|---|---|
| EllSeg DenseElNet, all datasets | [RSKothari/EllSeg](https://github.com/RSKothari/EllSeg/tree/8f1ea13336fa9c662403bd5823ac13cc6a6dd632) | `0b8d3f4c2c936de3950150d302f3608679874317862942e1ec53b21c33a03a1a` |
| RITnet, OpenEDS | [AayushKrChaudhary/RITnet](https://github.com/AayushKrChaudhary/RITnet/tree/6431c57ce7bf0eda935fb6178b926ae9440b50bf) | `02de259b8fe2e26e825101bb5d3f54b48a13a1a264eb0631c367db4596d0defa` |

Each repository includes its checkpoint and MIT license. The original checkouts
remain unmodified under ignored `dependencies/`. Both load every parameter with
`strict=True` and `torch.load(..., weights_only=True)`.

## Reproduce the runtime

Use the separate environment, **not the pipeline or face-tracking environments**.
The authors' old Linux/CUDA requirements cannot run on this Windows/Blackwell rig.

```powershell
git clone https://github.com/RSKothari/EllSeg.git dependencies/EllSeg
git -C dependencies/EllSeg checkout 8f1ea13336fa9c662403bd5823ac13cc6a6dd632
git clone https://github.com/AayushKrChaudhary/RITnet.git dependencies/RITnet
git -C dependencies/RITnet checkout 6431c57ce7bf0eda935fb6178b926ae9440b50bf
uv venv dependencies/ellseg-runtime/.venv --python dependencies/CorridorKey/.venv/Scripts/python.exe
uv pip install --python dependencies/ellseg-runtime/.venv/Scripts/python.exe torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python dependencies/ellseg-runtime/.venv/Scripts/python.exe -r scripts/data/r13-neural-runtime.txt
```

EllSeg's architecture needs two `np.int(...)` scalar casts changed to `int(...)`.
The adapter makes that substitution in memory when loading the original module;
it does not change weights, layers, loss functions or files in the checkout.
It directly invokes the encoder, ellipse regressor and decoder as the authors'
inference script does, avoiding construction of seven unused architectures.

EllSeg preprocessing uses grayscale, Lanczos width alignment to 320 pixels, then
vertical crop/pad to 240 and a per-image z-score. Integer crop/pad fixes the
upstream Python-3 slice-index bug. Half-pixel resize coordinates and the actual
rounded resize height are recorded in `native_to_model`; unsampled native rows
receive label -1. Pixel aspect enters this one resampling. Constant inputs abstain.
Softargmax uses the author's temperature 4 and normalized mesh; ellipse axes and
angle come from the pretrained regressor. Ellipses are stored as native boundary
curves and centers rather than incorrectly assuming circular native geometry.

RITnet preserves physical aspect while fitting into the author's 400x640 canvas,
with centered zero padding. Padding/crop context is an **unvalidated adaptation**
for performance footage. Its PIL grayscale conversion, gamma .8, CLAHE 1.5/8x8,
and normalization follow the author's `dataset.py`. No test-time augmentation,
threshold fitting, reporter-specific color calibration or fine tuning was done.

## What is reusable

`neural_iris_observations.py --shot <prepared shot> --review` uses the existing
dynamic native crop API and track metadata. It stores original scene-linear keyed
RGB, origin, EllSeg full-ellipse labels/boundaries, and RITnet visible-part proposals
under `relight/neural_iris_observations/`. No reporter coordinates or evaluation
labels enter this stage. Geometry/visibility flags remain false, so
`apply_eye_repair` returns exact source and scene-reflection permission stays false.
Its CLI requires `--shot`; omission lists shots.

**RITnet iris class is not clean iris texture.** It includes source catchlights.
**EllSeg iris class is not visibility.** It includes inferred occluded material.
Neither output is a removal mask or a donor-confidence certificate. The stage is
framewise inference seeded by existing face tracking, not a validated new temporal
tracker. Roll is recorded, but neural crops are not roll-normalized.

The adapter APIs scrub excluded native RGB before all resampling and learned
operations. That preserves a necessary boundary for future hidden-footprint tests.
This round performs no donor ranking, material correspondence or reconstruction.
The upstream face tracker is not included in its hidden-RGB invariance proof.

## Frozen evaluation and limits

The reporter-only audit scripts reuse round 10's native source cache and round 12's
sparse assistant labels. EllSeg ran on 1,200 source frames per eye. RITnet ran on
the 150-frame selected motion interval plus five non-overlapping annotated frames:
155 frames / 310 eye images. Both also ran on 150 keyed frames / 300 dynamic eye
crops. These are different crops and color paths; do not mix their metrics.

Sparse labels cover 12 eye images, eight open and four rejected/closed, with
estimated boundary uncertainty 1.5 native pixels. They are reused evaluation data,
not new independent human ground truth. No model/crop threshold was tuned to them.

| Sparse sampling | R12 candidate material | EllSeg full iris | RITnet visible iris proposal |
|---|---:|---:|---:|
| Clean iris points admitted / 40 | 26 | 40 | 30 |
| Noniris points admitted / 60 | 0 | 16 | 0 |

These masks have **different semantics**. This table demonstrates why raw EllSeg
output cannot replace a material-visibility mask; it is not a fair segmentation
accuracy ranking. None is certified by a sparse zero-false-positive count.
EllSeg predicts nonempty iris/pupil in all four closed-eye samples. RITnet outputs
zero iris/pupil pixels in three; source 79 left still has 10 iris and 4 pupil pixels.

On the eight open-eye images, median of per-eye pupil boundary medians changes
from 1.6472 native pixels (R12) to 1.0829 (EllSeg). Worst per-eye pupil p95 changes
5.3377 to 3.3405 pixels. Limbus medians change 0.8587 to 1.2465, while worst p95
changes 7.0812 to 2.5288. Thus improvement is not uniform. RITnet's source-923-left
pupil contour misses the sparse boundary by p95 5.5912 pixels; visual inspection
also shows missing iris/pupil regions. Its raster contour distances are not the
same continuous-ellipse metric. Detailed scores live in the shot's `logs/r13_*.json`.

## Checks and review

`verify_r13.py` checks upstream preprocessing parity (max abs 2.3842e-7), author
conic equations (max residual 7.8044e-8), multiple angle signs, square/anamorphic
mapping, two extreme hidden-RGB substitutions at each of two pixel aspects on one
real crop, constant-input abstention, all 300 exact keyed fallbacks, and 304 protected
hashes inherited from round 10. These adapter tests are not generalization or
positive reconstruction evidence. All six new Python files compile.

The three CRF 12 review videos run at the source's 60000/1001 fps:

- `r13_ellseg.mp4`: 1,200 frames, source / R12 raw candidate / EllSeg proposal.
  R12 masks are shown even on rejected candidates; they were never enabled repairs.
- `r13_ritnet.mp4`: 150 source-motion frames, source / visible-part proposal.
- `r13_neural_observations.mp4`: 150 keyed frames, source / combined proposals /
  exact source fallback. This demonstrates the reusable stage and retained glints.

All 1,500 encoded frames were decoded. All 150 ordered thumbnails for each short
video were inspected in aspect-preserving sheets. Full sparse RITnet boards for
79, 217, 521, 779, 923, 1115 and EllSeg boards 79/779 were inspected. The complete
1,200-frame EllSeg sequence was not visually inspected. No real-time playback or
artist approval is claimed. `logs/r13_visual_review.json` records that scope;
verification itself leaves visual status pending so re-runs cannot imply approval.

Production keys, source, ratio compositor, shot configuration and corneal renderer
are unchanged. Original catchlights, crown sheen and lip gloss remain unresolved.
