# Round 19 UnReflectAnything availability and highlight-head evaluation

Added after the artist requested a current research check and explicitly included
drums/guitars and other reflective materials in the intended scope. Written before
UnReflectAnything inference. This supplements, and does not change, the frozen
local-motion protocol or its results.

- Verify the downloaded author full checkpoint at HF revision
  `ed889890c625698c8665d872fc02987bafd1b134`, SHA256
  `12c0cf4518a39325b7bfe2bed04a96395189f2643c2f90b7b79360a659dc739b`.
  Author code revision `363ebb5c87d9f4f49b63f744fded8d3786a051b3`.
  Use the embedded training config, not the stale packaged token-prior weight.
  Load all 601 state entries strictly, verifying exact tensor equality. A separate
  torch 2.9.1/cu128 environment uses the author's pinned Transformers commit.
- Load with weights_only=True and the explicit DotMap config allowlist. Only
  change runtime file paths and disable compile using the author's inference
  helper. The local encoder mirror revision is
  `f43714afb20addc93acad542bf6ea0376b6f7900`; full weights supersede its initialization.
- Evaluate the trained highlight head on both unchanged 64x64 eye crops for all
  source frames 950..999 (100 inputs). Keep author preprocessing, 896px model
  resolution, float32 and eval mode. No thresholds/crops chosen after inference.
  Resize soft highlight probabilities back to native using bilinear align_corners=False.
- This is a highlight proposal diagnostic, not a diffuse reconstruction test.
  Use only encoder and highlight head at inference, matching those operations
  from the full model. Do not invoke token inpainting/diffuse decoding or use
  synthesized performer RGB. No removal, clean-donor certification or replacement.
- Report probability at all 12 reused round-14 reflection anchors and 30 iris
  anchors, with the author's 0.3 threshold, and predicted native area on the
  source-labelled nearly closed frames 981/982. Sparse point scores are diagnostic
  and not complete reflection masks; iris annotations do not certify clean radiance.
- Save source hashes, raw model maps, native maps, model/config/code provenance,
  peak CUDA allocation and runtime. Repeat source 977, both eyes, for reproducibility.
- Produce a source-fps video: original eyes left, soft probability maps right;
  label predictions unvalidated. Decode all frames; inspect ordered contact sheets.
  Existing exact fallback and protected-file verification must still pass.
- Neither this test nor published general highlight-removal results establish
  performance on drums, guitars, polished metal or arbitrary footage. The next
  material benchmark must include these when prepared footage is available.
