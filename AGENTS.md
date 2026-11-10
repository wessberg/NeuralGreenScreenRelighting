# Working guide

This is a generic actor-relighting library/pipeline. Start with README.md,
docs/ARCHITECTURE.md and docs/INTEGRATION.md. Historical research context is preserved
in docs/history/development-guide.md, not deleted.

## Current direction

- CorridorKey 2.0 is the default keyer; source revision is in dependencies.lock.json.
  scripts/key_plate_v1.py is the historical keyer. Never retune the accepted shot's
  key thresholds or overwrite its keys to compare a new model.
- EyeClean is the removal backend. Use float patches before the ratio and gate scene
  reflections on successful per-eye removal. Keep exact source fallback on abstention.
- Neural normals drive shading; proxy geometry and Blender supply occlusion/shadows.
  The performer remains the photographed plate.
- Geometry/compositor scripts still require a source checkout; src/ngsr is the
  reusable adapter/API boundary. Keep engines out of the host interpreter.

## Accepted appearance and rejections

Round 44's r44_m50_s05 is the accepted picture; r42_wet supplies the accepted eyes.
Meso strength 0.5 and skin specular 0.5, roughness 0.55, F0 0.028, plate texture modulation.
It is improved, not fully photorealistic. New dependencies do not inherit that approval.

Round 45's hair removal/replacement was entirely rejected, including attenuation alone.
Leave hair alone. Do not reattempt it without asking the artist. A positive highlight/light
correlation is diagnostic, never an acceptance criterion. Lip work also requires its own
visual review; it is outside release preparation.

When an effect is too strong, decompose before tuning. Smooth neural normals plus a
physical-strength lobe produce a smooth Barbie-like band. Texture modulation is a heuristic,
not recovered microstructure. Do not claim the appearance bar has been met numerically.

## Engineering

- GPU first for per-pixel maths. Run neural inference and Blender sequentially.
- Verify axes/handedness/signs with independent controls in several directions.
  World normals use (c0, -c2, c1), recorded by the convention validator.
- Every source is explicitly colour-managed/desqueezed upstream. The current camera-log
  ingest is incomplete; do not feed V-Log as if it were linear/sRGB.
- Source footage and accepted baselines are immutable. Use fresh shot/run directories.
- No on-set chart/reference is a prerequisite. Infer source lighting when possible;
  label camera/material assumptions.
- Blender 5.2.1: persistent data OFF; emissive meshes visible to camera rays can
  contaminate irradiance passes. Use actual Cycles filter settings, not render.filter_size.
  Multilayer EXRs require Blender extraction; OpenCV does not read them.
- Review changed eyes source/result in the SAME tile, at zoom and face scale, over time.
  Numerical checks support artistic review; they do not replace it.
- Test contracts without requiring weights; run the reporter smoke for integration changes.
- Dependencies and weights have separate licences. Do not publish media or rewrite history
  as part of ordinary code changes. docs/RELEASE.md tracks publication prerequisites.
