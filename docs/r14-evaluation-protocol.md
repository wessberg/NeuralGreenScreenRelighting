# Round 14 frozen evaluation protocol

Written before round 14 model results or fresh annotation boards are inspected.
No fitting of thresholds or selection using the round 12 sparse labels. This is
a new evaluation on the **same reporter clip**, not independent footage.

- Source targets: frames 950–999 inclusive, both eyes. Correspondence tests on
  every second frame (950, 952, …, 998). Source donors: 0–699 at stride 4,
  excluding frames within 8 of the old annotation frames 79, 217 and 521.
  No target frame is a donor. Donor spacing and temporal guard remain 4 and 3.
- Fresh sparse semantic annotations: frames 959, 977, 995, both eyes, made on
  source-only native grids before viewing predictions. Assistant annotations,
  not human ground truth. Record unequivocal iris/noniris points and reflection
  points; ambiguity and boundary uncertainty must remain explicit.
- Test holes: five native pixels (radius-one cross), centered at 0.7 times the
  coarse landmark radius at 60 and 120 degrees. Locations depend only on the
  frozen upstream track, never neural acceptance or hidden RGB. Report every
  attempted hole, including abstentions and holes outside inferred iris.
- Rerun both pretrained models with target hidden RGB scrubbed before any
  preprocessing. The face tracker and crop seed remain fixed and outside this
  invariance claim. Test black and magenta hidden substitutions.
- Geometry: recover native ellipse axes from EllSeg curves; concentric
  pupil/limbus sampler is an approximation. Gate pupil-center displacement at
  2 native pixels. RITnet supplies a distinct visible-part proposal; intersect
  iris proposals and erode by one pixel. Require >=40 candidate iris pixels,
  >=12 RITnet pupil pixels, pupil ratio in [0.15, 0.75]. These are experimental
  abstention rules, not calibrated probabilities or established safe thresholds.
- Compare fixed material coordinates with fitted torsion using the inherited
  search [-0.18, 0.18] radians, 19 steps. Fit and validation occupy separated
  angular sectors. Exclude any native donor interpolation footprint shared by
  fitting, context validation, or the query. Rank donors using fitting data only.
  Both arms start with the same fixed-map shortlist (24); retain at most 12.
- Inherited context gate: training and validation MAE <=3 display code values,
  wrong-rotation (+0.35 radian) control worse by >=1 on identical pixels,
  at least 8 common control pixels. Reconstruction needs >=3 spaced donors,
  donor MAD <=2. Report paired support, gate coverage, errors and negative
  controls; agreement alone never authorizes repair.
- RITnet does not mark reflections. Its iris proposals remain potentially
  contaminated donors; the new sparse labels assess this directly. No new
  brightness/contrast detector will be substituted for that missing evidence.
- Probe fresh manually marked reflections with the same correspondence path;
  there is no hidden clean truth at a real glint. Pupil-crossing or ambiguous
  material must abstain. Sparse reflection points are not complete removal masks.
- Exact source-keyed fallback remains mandatory. No production repair or scene
  reflection may be enabled without independently validated geometry, material,
  correspondence and complete source-removal coverage. Keep Cycles corneal pose,
  lid/lash/brow and incident-light occlusion as explicit downstream blockers.
- Produce 50 ordered diagnostic review frames at shot fps, source on the left,
  proposal/gate overlay in the middle, unchanged source on the right. Decode the
  video and inspect source/overlay boards and temporal contact sheets. Do not
  present a fallback video as a reflection fix.
