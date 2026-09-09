# Round 18 frozen material-versus-reflection motion protocol

Written before new inference or error inspection, 2026-09-09.

- This is a synthetic correspondence stress test, never a real clean-radiance
  test. Use the observed 64x64 source-977 eye crops and the first and fourth
  round-14 iris anchors on each eye (four anchors). Original photographed
  reflections remain part of the translated base image. Neither that image
  nor the sparse labels are certified reflection-free iris material.
- For each anchor, independently build 24 frames in each of +x and +y.
  Translate the entire observed crop by floor(t/4) native pixels, using integer
  indexing and reflected borders. Score interior points only. This known
  whole-image motion does not simulate a rotating cornea or independently
  moving lids. Pixel aspect is the reporter's square-pixel cache space.
- Pair an observed-base-only control with a synthetic additive-reflection
  sequence. Decode cached display sRGB to linear, add a neutral compact softbox
  of peak 0.6, then encode to display sRGB, clipping only for model input.
  Its centre is q + 3*axis + (12-t)*axis; its support has half-width 3 and
  half-height 3 native pixels, separable raised cosine. Thus reflection and
  material coincide at t=12, but move in opposite directions. Report clipping.
  Do not tune shape, brightness, motion or points after seeing predictions.
- Track one query at a time in both sequences: t=0 is an observed-iris anchor
  moving toward contamination, and t=12 is a contaminated anchor. The query
  follows the same known base-image point in both cases. CoTracker3 and RAFT
  use unchanged released adapters and weights. Query frames are excluded from
  scores, because CoTracker forces them. Direct RAFT evaluation frames are
  0,4,8,12,16,20,23, minus the anchor frame; CoTracker is compared at exactly
  these six frames, with full temporal tracks retained for review.
- Scrub the complete native 2x2 query footprint before every resize, including
  reverse RAFT input. Recompute each full inference with black and magenta
  substituted at those contributors; require exact saved arrays. The rest of
  the reflection remains visible, intentionally testing reflection lock-on.
  This is a coordinate experiment, not hidden-context photometric validation.
- Report all attempts, endpoint error to the known material path, acceptance
  (CoTracker visibility AND native bounds; RAFT cycle <=0.5 px AND bounds),
  accepted errors, and accepted errors >1 native pixel. Reflection lock evidence
  means accepted AND reflection EPE <=1 AND material EPE >1, on frames where
  material/reflection centres differ by >=3 pixels. Also report rejection and
  all-attempt errors, so abstention cannot appear as accuracy. Compare the
  paired base-only control; do not interpret a false track as reflection lock
  unless the reflection-distance criterion actually passes.
- No donor RGB reconstruction, photometric gate, revised real control design,
  removal, scene replacement or production change. This round first tests
  the identity assumption needed to choose a useful next real experiment.
  The eight-context / three-spaced-donor / MAE / MAD / wrong-map gates stay
  unchanged. All-arm context exclusions remain necessary before real scoring.
- Independently verify synthesis, coordinates, cycle and metric counts; audit
  protocol/source/model/code provenance, previous protected hashes and all 300
  exact keyed fallbacks. Render a labelled temporal review of all eight motion
  sequences and both anchor directions. Clearly distinguish synthetic motion,
  model acceptance, known material truth, and reflection centre. Base control
  appears on the left. Decode and inspect all frames and selected full boards.
- Write round18.md with measured limitations and a round-19 starter prompt.
  Full real reflection extent, pupil crossing, same-point clean donor identity,
  corneal pose, lid/lash/brow occlusion, hair sheen and lip gloss remain open.
