# Round 15 frozen dense correspondence protocol

Written before RAFT inference/results. This is a bounded diagnostic, not a repair.

- Component: torchvision 0.23 RAFT large, explicit `C_T_SKHT_V2` released weights,
  12 updates, float32 evaluation. Existing isolated ellseg runtime; no upgrades.
  [Official model documentation](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.optical_flow.raft_large.html)
  and [preprocessing example](https://docs.pytorch.org/vision/0.23/auto_examples/others/plot_optical_flow.html).
  RAFT estimates two-frame displacement, not material identity or visibility.
- Reuse all 30 round 14 source-only iris points (959, 977, 995, both eyes).
  These labels and this clip are seen data, with 1.5 native-pixel annotation
  uncertainty. No new/unseen-data or certified-clean-material accuracy claim.
- Fixed six same-eye donors per target: offsets -12, -8, -4, +4, +8, +12.
  This evaluates nearby temporal correspondence, not round 14's distant pool.
  Do not numerically rank rounds with different donor pools.
- Mask all five iris points together before resize, normalization, either flow
  direction, or target EllSeg/RITnet inference. Fixed upstream crop seeds are
  outside the hidden-RGB guarantee. Test black and magenta substitutions with
  complete recomputation. Inference sees surrounding target RGB and donor RGB;
  this is frozen pretrained inference, not an independently held-out context fit.
- Input crops are native display RGB; honor supplied pixel aspect when resizing
  to physical proportions with both dimensions >=128 and divisible by 8.
  Resize bilinearly without antialiasing per torchvision example. Return native
  displacement and sample only original native donor RGB for estimates.
- Compare RAFT mapping with zero displacement in the same fixed tracked crops.
  Also compare +3 native-pixel horizontal wrong-map control on identical context
  pixels. No per-case optimization, shortlist, donor ranking or torsion search.
  Across ALL evaluated maps (RAFT, fixed, and their wrong-map controls), exclude
  donor context footprints intersecting any query footprint. Retain footprint
  arrays and assert exclusions. Existing round 14 search-isolation regression
  remains required. These controls do not imply independence from neural image
  inference, whose full context is explicitly disclosed above.
- Report raw in-bounds support separately from candidate support (complete native
  bilinear footprint in the eroded combined donor iris proposal), proposal gates,
  forward/backward residual, and photometric errors. Target candidate gates are
  scored but do not hide raw known-iris attempts from the denominator.
- Experimental context gate: >=8 identical candidate context pixels, MAE <=3
  display levels, wrong-map MAE worse by >=1. Query round trip <=0.5 native px;
  aggregate requires >=3 spaced donors and MAD <=2 display levels. Both source
  and donor candidate gates must pass for gated donor acceptance. These are
  uncalibrated diagnostics, never authorization to replace an eye. All 30 target
  points, including unsupported points, remain in the denominator. Report paired
  errors on identical native samples; no success claims from agreement alone.
- Reused blink stress sequence: source 950-999, both eyes, anchor the five reused
  iris points at 977 and map anchor directly to each frame. Inspect source 984
  left (known false candidate) and 985 both (known nearly closed failures).
  Evaluate forward/backward consistency and neural candidate masks separately.
  No true material track/occlusion ground truth is available; a passing round
  trip through closure must not be called visibility. Review every ordered frame.
- Probe all 12 reused sparse reflection points separately, with hidden RGB
  excluded. Hidden clean truth is unknown; never score original reflection RGB
  as clean truth. Their full extent and possible pupil crossing remain blockers.
- Keep exact source-keyed fallback; verify all 300 existing eye crops and 304
  protected hashes. Production compositor remains off for removal/replacement.
  Corneal pose, lid/lash/brow visibility and incident-source occlusion remain
  downstream tasks before Cycles replacement. Hair/lip removal remains unresolved.
- Produce a 50-frame source / RAFT tracking diagnostic / exact source fallback
  review at shot fps, CRF 12; fully decode and inspect ordered contact sheets.
  Write round15.md with measured results, limits and next-round starter.
