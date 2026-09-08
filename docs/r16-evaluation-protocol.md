# Round 16 frozen visibility and donor-material protocol

Written before CoTracker inference. Source-only boards for 974, 978, 981, 982,
984, 985, 990, 994 were inspected before writing labels or viewing new predictions.

- Use authors' released CoTracker3 scaled offline checkpoint and unmodified
  predictor at repository commit 82e02e8029753ad4ef13cf06be7f4fc5facdda4d.
  [Official implementation](https://github.com/facebookresearch/co-tracker/tree/82e02e8029753ad4ef13cf06be7f4fc5facdda4d).
  Predictor uses six iterations, a 6x6 support grid, visibility >0.9 and forces
  query-frame coordinates/visibility. Exclude the query frame from all scores.
- Reuse round 15's 950..999 eye crops and ten iris queries at 977. Compare
  unmasked CoTracker offline/backward tracking with saved unmasked direct RAFT
  maps, <=0.5 px cycle, and the same saved neural material proposal. These are
  seen footage and reused query annotations, not new generalization evidence.
- New source-only region annotations on eight frames per eye are independent of
  model output, NOT an independent annotator or certified reflection-free truth.
  Store explicit conservative iris, pupil, sclera, skin/lash, and reflection
  regions. Unlabelled pixels and a 1.5 px boundary band are UNKNOWN, not negative.
  A complete 2x2 native bilinear footprint must fit inside one region to classify.
  Visible iris compatibility does not prove the same material point or absence
  of a weak reflection. Reflection pixels have no clean RGB ground truth.
- At 981 and 982 both eyes have no resolvable iris in source-only inspection:
  report this closure annotation separately as a real visibility stress test.
  It is limited by blur and source resolution, not a calibrated ground-truth
  visibility dataset. Other narrowed frames remain unknown for track occlusion.
- For each arm report all 80 labelled-frame query attempts, visible predictions,
  iris-compatible footprints, explicit incompatible footprints (with classes),
  unknowns, and coverage. Compare CoTracker visibility alone, RAFT cycle alone,
  RAFT cycle+proposal, and CoTracker visibility+the identical proposal. Do not
  turn an unknown into a clean donor. No photometric/context gate is relaxed.
- Add a controlled temporal occlusion test on observed eye RGB: 24 frames,
  horizontal +/-3 and vertical +/-3 native px translations, opaque rectangle
  x=12..51,y=12..51 in frames 8..15, four interior iris queries at frame 0.
  Known geometric tracks and full occlusion supply synthetic truth. Score
  visible recall, occlusion false positives, endpoint error, and recovery
  separately; do not pool synthetic and real results. No generated performer RGB.
- Reusable adapter accepts native RGB video, arbitrary t/x/y queries and pixel
  aspect. Letterbox into the model canvas to preserve physical proportions,
  using align_corners=True coordinate transforms and return native coordinates.
  Verify +/-x,+/-y and both square and anamorphic aspects analytically and via
  the controlled video. Padding and incomplete native footprints are invalid.
- Scrub exclusions BEFORE any model resizing. Recompute both real eyes with
  black and magenta substitutions at anchor query pixels; compare masked
  coordinates/visibility exactly. This only guarantees excluded RGB invariance;
  the fixed upstream crop tracker and surrounding image context remain visible.
  Unmasked visibility evaluation is not a withheld-RGB radiance experiment.
- Independently resample all saved tracks from native donor RGB and audit full
  footprints against frozen source labels. No radiance truth under reflections,
  no donor ranking, no fitted context test, and no reconstruction in this round.
  Preserve prior native-footprint/search controls and exact keyed fallback.
- Produce 50 ordered review frames at source fps, CRF 12: source, RAFT diagnostics,
  CoTracker diagnostics, exact fallback. Decode all frames and inspect contact
  sheets and closure boards. No visual or artist approval implied by decoding.
- Keep production repair and scene replacement disabled. Verify 300 keyed eye
  fallbacks and 304 protected hashes. Record protocol/labels/code/checkpoint hashes.
  Complete removal, corneal pose, pupil crossing, camera and incident-light
  lid/lash/brow occlusion, hair sheen and lip gloss remain unresolved.
