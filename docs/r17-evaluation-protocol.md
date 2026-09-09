# Round 17 frozen reflection-material and context-capacity protocol

Written before round 17 inference or photometric scoring, 2026-09-09.

- Reuse all 12 round 14 reflection points at 959, 977, 995, both eyes.
  Run each query separately with its complete native 2x2 footprint scrubbed
  before resizing in both pretrained adapters. CoTracker sees 950..999;
  RAFT maps the same target to the eight round 16 labelled donor frames:
  974, 978, 981, 982, 984, 985, 990, 994. Keep all 96 attempts per arm.
  CoTracker's forced query frame is never a donor or scored observation.
- Keep the released round 15 RAFT and round 16 CoTracker settings. Compare
  in-bounds CoTracker visibility and RAFT cycle <=0.5 native px, separately
  and intersected with the SAME saved neural donor proposal. No claim that
  cycle is visibility or that a reflection-anchored track follows iris texture.
- Classify complete donor footprints with UNCHANGED round 16 source regions
  and 1.5 px uncertainty. Unknown stays unknown. Reflection/pupil/skin/sclera
  are explicit incompatible material for an iris donor; iris compatibility
  does not certify clean radiance or same-point identity. No RGB error against
  the original catchlight: clean truth there is unavailable. No donor ranking.
- Independently inspect the existing source-only boards for 959/977/995, then
  freeze additional conservative iris regions in r17_context_labels.json.
  These are assistant annotations on seen images, before this round's new
  predictions; NOT unseen or independent human truth. Do not alter r16 labels.
- Preflight all 30 reused iris queries AND 12 reflection probes, one hidden
  query at a time. Context uses full native footprints inside frozen target
  iris interiors. Fixed donors for this preflight are 974, 990, 994 (same eye).
  These three are >=4 source frames apart. Exclude target query footprint and
  donor query footprints across fixed and +3px horizontal wrong-map controls.
  Count context common to fixed and wrong maps using frozen donor iris labels.
  This is an UPPER BOUND for a later intersection with RAFT/CoTracker maps and
  their wrong controls. It is specific to this control design and labels, not
  an impossibility bound on the footage. Save capacities before inference;
  do not inspect RGB errors to choose cases, labels, donors or thresholds.
- The unchanged gate needs >=8 common context samples PER donor, MAE <=3
  display levels, wrong-map margin >=1; >=3 spaced donors and MAD <=2.
  A case with fewer than 3 capacity-eligible donors is insufficient: do not
  score a photometric comparison or infer success from agreement. If any
  case clears preflight, stop the scoring phase until full all-arm footprints
  and masked context inference are implemented; capacity is not a gate pass.
- Recompute each reflection query with black and magenta substitutions at
  every excluded target contributor. Require exact coordinates, visibility,
  native validity, both RAFT fields and material outcomes. Fixed upstream
  crop/landmark seeds are outside this guarantee; surrounding input RGB is
  visible to frozen models. This is not held-out-context model validation.
- Independently enumerate native sampling contributors, check source hashes,
  previous code/evidence hashes, the round 14 search isolation regression,
  and all 300 exact keyed-eye fallbacks / 304 protected files.
- Produce 50 ordered source-fps review frames, showing source, masked RAFT
  and CoTracker reflection queries anchored at 977, and exact source fallback.
  RAFT is shown only at its eight evaluated frames (no interpolated tracks).
  Label no removal/replacement and forced query visibility. Decode all frames,
  inspect ordered contact sheets and full boards; do not imply artist approval.
- No production removal or scene reflection, generated performer RGB, key
  retuning or relaxed gates. Full reflection extent, pupil crossing, clean
  same-point donor identity, real corneal pose, camera and incident-source
  lid/lash/brow occlusion, hair sheen and lip gloss remain open. Write round17.
