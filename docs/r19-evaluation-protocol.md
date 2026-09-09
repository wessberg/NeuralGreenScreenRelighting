# Round 19 frozen local-motion and registered-context protocol

Frozen before new synthesis scores, neural inference or real capacity counts,
2026-09-09. No production repair or scene replacement is authorized by this test.

## Local motion experiment

- Reuse all eight round-18 source-977 point/axis definitions, both anchor
  directions (0,12), both base/added-reflection conditions, released models,
  inference adapters, evaluated frames, hidden-footprint substitutions and
  acceptance/error/reflection-lock definitions unchanged. No model tuning.
- Change only the base motion: within a stationary circular aperture centered
  on the original query, observed RGB translates by floor(t/4) along +x/+y.
  Radius <=8 native pixels has exact integer-translated source pixels; radius
  >=12 has the exact stationary source. Between them, a raised-cosine weight
  blends translated and stationary observations in linear light. All query
  2x2 footprints remain in the exact-motion interior. This is a local observed
  eye-texture patch, NOT a segmented anatomical iris or a rotating cornea.
  Its artificial fixed boundary is an additional cue and a stated limitation.
- Added reflection shape, amplitude, linear-light addition, trajectory and
  clipping are exactly round 18. Original photographed highlights remain in
  the base. The synthetic appearance is not clean material ground truth.
- The stationary-surround predictor remains at the anchor coordinate. Report
  its EPE and accepted model lock to this coordinate (material EPE >1,
  stationary EPE <=1, truth separated >=3), alongside reflection lock.
  This distinguishes following the background from following local texture.
- Reuse the hashed round-18 whole-crop results as the paired global-motion
  reference; do not rerun them or present them as new evidence. Compare all
  attempted frames, report base-control failures and correlated sequence IDs.
- Generate eight seconds of labelled temporal review, base on left, both
  models and anchors; decode all frames and inspect ordered sheets/full boards.

## Real context preflight (RGB free)

- Keep all 42 round-17 queries, frozen source-only polygons, 1.5px boundary
  uncertainty, and donors 974/990/994. No new annotations or model predictions.
- Replace the unregistered fixed baseline with an integer translation from
  target to donor iris-mask centroid (round to nearest integer, ties to even).
  This is a geometry-only proposal, not inferred same-point correspondence.
  Compare it with its unchanged +3px horizontal wrong-map control. This is
  an alternative design; do not claim it is the old all-arm intersection.
- Enumerate complete native 2x2 context footprints in target and both donor
  maps, excluding the target query and BOTH donor query footprints. Count
  their common capacity. Learned maps and their query/context exclusions
  would still need intersecting; this is an optimistic upper bound only.
- Independently enumerate contributors and verify centroid signs using four
  signed synthetic mask translations. No real RGB scores this round, even
  if capacity is adequate. A later round must first demonstrate that this
  control rejects known wrong correspondences and implement all learned arms.
  Capacity alone cannot establish discrimination or authorize photometry.
- Preserve >=8 identical context samples, >=3 donors spaced >=4 frames,
  MAE <=3 display levels, wrong-map margin >=1 and MAD <=2. No relaxed gates.

Independently check synthesis, exact query contributors, stationary exterior,
linear transfer/addition, clipping, coordinates, cycle, bounds and all counts.
Require black/magenta full inference recomputations to match saved array hashes.
Verify model/code/source/protocol provenance, prior code and protected files,
all 300 exact keyed-eye fallbacks, and the search-isolation regression.
Write round19.md with limitations and a round-20 starter prompt. Real material
identity, full reflection extent, pupil crossing, corneal pose and lid/lash/brow
occlusion still block coupled reflection removal/replacement; hair/lips remain open.
