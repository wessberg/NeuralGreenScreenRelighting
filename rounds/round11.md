# Round 11 handoff — temporal iris core and accepted source-keyed fallback

**Date:** 2026-09-08. Starts from round 10 and round-4-equivalent production defaults.
**No real-footage reflection repair was delivered.** A reusable reconstruction core now
has positive known-truth synthetic controls and an exact source-keyed fallback. Automatic
real pupil/iris segmentation, material tracking, and compositor integration remain unfinished.
Original reporter catchlights, crown sheen and lip gloss remain unresolved.

## 1. Artist decisions — preserve these in future rounds

The artist corrected two assumptions during this round:

- This is an **input-agnostic tool**, not a bespoke reporter repair. Failure on one clip
  under one alignment/mask family is not a general impossibility result.
- The artist specifically wants reconstruction using iris details visible in other frames.
  This remains a valid direction. Motion of the reflection relative to actual iris material
  is what supplies evidence; motion in image coordinates alone does not establish coverage.
- **For frames/angles where iris detection is unreliable, keep the source keyed eye and
  its existing catchlight, if any.** The artist explicitly accepts this fallback. We also
  retain source when material correspondence or donor evidence is unreliable.

Do not demand successful removal at every angle before considering useful reliable-view
repairs. Do not replace a failed detection with an invented eye. Work per eye, preserve
fallback frames exactly at the keyed-foreground stage, and inspect the transitions.

The initial lip investigation was stopped following this steering. No lip implementation,
repair, or new lip claim is part of the round's delivered work.

## 2. Delivered code and boundaries

`scripts/temporal_iris.py` is a library with no reporter identifiers, camera constants,
skin/iris colour thresholds, source-rate assumptions, or model downloads:

- `IrisGeometry`: projected concentric pupil/limbus ellipses, image-plane rotation,
  pupil dilation and iris texture torsion. Converts native pixels to shared iris material
  coordinates and back; the pupil itself is excluded from iris sampling.
- `reconstruct`: samples each material point from eligible donor frames. Requires all four
  native bilinear contributors to pass supplied material/visibility/reflection exclusions.
  Uses at least three donors, selected at least four frame IDs apart, excluding target +/-3.
  Returns RGB, coverage, donor agreement, counts and per-sample donor provenance.
- `apply_eye_repair`: requires independently supplied reliable detection and material
  correspondence flags, complete support and agreement throughout the requested repair
  mask, and finite estimates. Otherwise returns the exact original keyed image. It never
  fills unsupported pixels spatially. Pixels outside the repair mask remain exact.
- `repair_strength`: three-frame taper inside trusted runs; unreliable frames stay exactly
  zero. This reduces correction switching but is not proof of invisible transitions.

The same RGB units/working space must be used for donors and target; the module does no
colour conversion. Agreement tolerance is a **required argument in caller RGB units**;
the diagnostic explicitly supplies 2 display levels. A future scene-linear caller must
supply its appropriate validated tolerance; do not reuse 2 as a scene-linear threshold.

This is a **sampling/application core, not an automatic eye-repair stage**. Reliability flags,
boundary geometry and masks are inputs; the library cannot certify them. It is not imported
by `composite_ratio.py`, and no new production flag was enabled.

The linear pupil-to-limbus coordinate idea follows iris normalisation work. The model is
an approximation: off-angle refraction, nonconcentric boundaries and nonlinear iris
deformation are not solved here. See the primary papers
[Effect of pupil dilation on off-angle iris recognition](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-28/issue-3/033022/Effect-of-pupil-dilation-on-off-angle-iris-recognition/10.1117/1.JEI.28.3.033022.full)
and [Dilation Aware Multi-Image Enrollment for Iris Biometrics](https://www3.nd.edu/~kwb/Ortiz_Bowyer_IJCB_2011.pdf).
Their recognition results do not validate our photographic reconstruction.

## 3. Known-truth reconstruction controls — positive evidence, explicitly synthetic

`scripts/audit_r11_atlas.py --shot <dir>` generates three 48-frame tests. Procedural iris
textures, pupil/limbus geometry, eyelid aperture and reflection masks are **known by
construction**. These do not test real-image segmentation or show a reconstructed reporter.

Both arms use the same native RGB, masks, target holes and temporal exclusions. The control
holds pupil ratio and texture torsion fixed while still using known ellipse position, axes
and image rotation. The other arm uses their actual changes. There is no brightness-minimum
selection, gain fit, colour fit, or generative performer in the sampler.

| Synthetic condition | Requested native reflection pixels | Supported: fixed / varying | Matched per-frame MAE median: fixed / varying | MAE p95: fixed / varying |
|---|---:|---:|---:|---:|
| Brown, nominal 13 px radius, moving reflection | 941 | 932 / 933 | 5.103 / 1.755 | 6.433 / 2.195 |
| Blue, nominal 30 px radius, moving reflection | 5,069 | 5,042 / 5,057 | 4.861 / 0.514 | 6.209 / 0.561 |
| Brown, nominal 13 px radius, reflection fixed to material | 1,059 | 77 / **0** | No matched support | No matched support |

RGB errors are display-channel absolute differences on a 0–255 scale, computed before
review encoding. Medians/p95 weight supported frames equally; support counts are summed
native query pixels. The larger and smaller tests differ in colour and radius, so this is
not a controlled resolution-only comparison.

The correct model abstains completely in the fixed-reflection negative control. The wrong
model produces 77 spurious samples, **57 even passing donor agreement**. Their conditional
per-frame median error is 7.050 display levels. Thus donor agreement is not a substitute for
verified material correspondence; matching the wrong place can produce confident mistakes.

Checks pass: four independent signed rotation/axis checks (max error <3.1e-7 px), coordinate
round trips (<1.5e-14 px), native four-contributor exclusion, missing-evidence abstention,
and twelve complete target-RGB perturbations. Those perturbations hold known geometry and
masks fixed; they do not test a future detector's independence from hidden source pixels.

## 4. Fallback and temporal controls

The 48-frame small synthetic case forces failed detection at 12–17 and 36–41, then combines
that with complete donor agreement. Result: **21 repaired frames, 27 exact source fallback
frames**. No pixel outside the requested mask changes. Bad detection always returns source.

The maximum adjacent repair-weight change is 0.333334. Maximum adjacent change in
whole-image mean correction decreases from **0.261222 to 0.087074 display levels** with the
taper. This diluted whole-image diagnostic establishes a reduced correction switch, not
perceptual invisibility or a real-footage flicker result. The fallback still visibly retains
the reflection, as requested; partially weighted reliable frames retain some of it too.

On all **150 actual keyed reporter foregrounds**, both failed-detection and failed-material-
correspondence gates were tested: **zero changed channels across 1,866,240,000 comparisons**.
Every real frame was deliberately sent to fallback because new real geometry/correspondence
has not been validated. This is a preservation test, not automatic identification of good views.

Additional branch tests cover incomplete support, failed agreement and nonfinite estimates;
all return exact source. A positive valid estimate changes only its requested mask.
The 304 protected-file hashes from round 10 still match, including source plate, shot config,
cached landmarks, production compositor, 150 keyed foregrounds and 150 baseline composites.
Both new scripts compile. Production defaults remain unchanged.

## 5. Review evidence and reproduction

Files under `music_videos/reporter_test/shots/reporter_a/final/`:

- `r11_synthetic_correspondence.mp4`: 144 frames at 24 fps; input/known truth above,
  fixed/varying model below. Purple means unsupported evidence, not a repaired colour.
- `r11_synthetic_fallback.mp4`: 48 frames at 24 fps; input/fallback-aware result above,
  truth/untapered switching control below. Detection failures are synthetic injections.
- `r11_native_fallback.mp4`: 150 frames at source 60000/1001 fps; original keyed eyes left,
  exact fallback right. **There is deliberately no visible improvement in this video.**

All are CRF 12 review encodes and were decoded completely: **342 frames**. Every frame was
inspected in fifteen ordered contact sheets (24 thumbnails per full sheet, 240x168 each).
The contact-sheet resize distorts aspect and loses detail; full PNGs at synthetic comparison
51, fallback 8/12 and native fallback 79 were also inspected. This was ordered-frame visual
inspection, not real-time playback or artist approval.

Observed: the synthetic moving-reflection reconstruction follows the known texture with
some native-pixel interpolation loss; the fixed-reflection negative control stays purple;
the fallback demonstration restores the original bright reflection on rejected frames;
the native reporter comparison preserves the original glints through blinks and head motion.
The synthetic diagrams are technical test images, not photorealism demonstrations.

Before the new tests, round 10 source 779's mask counterexample and a native full-source eye
survey at source 0, 100, ... 1100 were inspected. The survey shows pose/reflection changes;
it does not establish which covered material points have clean donors.

Evidence: `logs/r11_atlas_audit.json`, `logs/r11_visual_review.json`, each video's encode
sidecar, and regenerable `composite/r11_atlas/` images. `review_video.json` now describes
the latest native fallback encode. Re-running the audit marks visual inspection pending.

```powershell
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/audit_r11_atlas.py --shot music_videos/reporter_test/shots/reporter_a
```

No on-set references, key threshold changes, generative face replacement, added corneal
render, or persistent Cycles data. Hair and lips received no treatment in the delivered round.

## 6. Starter prompt for round 12

> Read round11.md. The artist wants input-agnostic temporal iris reconstruction and explicitly
> accepts keeping the original keyed eye/catchlight for unreliable detection or angles.
> The generic sampler and exact fallback exist, but positive reconstruction evidence is only
> synthetic with known geometry/masks. Do not run another synthetic demonstration and call it
> real progress on the reporter. Build and validate native-image pupil/limbus visibility and
> actual iris material correspondence on good real views, allowing rejected views to fall
> back. Preserve native hidden-footprint controls and test the detector's role in leakage.
> Wire trustworthy observations into the shared material-coordinate sampler and inspect an
> actual repaired-versus-source video, including repair/fallback transitions. Do not use the
> old MediaPipe aperture as a semantic iris label or donor agreement as a correspondence
> certificate. The sampler excludes the pupil; any reflection crossing it needs its own
> validated treatment or whole-eye fallback. Corneal source/lid/brow/lash occlusion remains
> necessary before synthetic replacement. Keep production unchanged until real-view evidence
> supports enabling a repair. Write round12.md with a starter prompt for round 13.
