# Round 8 handoff — isolate added gloss; temporal iris recovery fails a blind check

**Date:** 2026-09-08. Starts from round 7's verified round-4-equivalent defaults.
**No visual fix for the three original complaints was delivered.** This round produced
controlled comparisons and a new recovery test, and rejected the tested shortcut before it
could alter the performer. No photorealism row becomes green. Artist verdict remains pending.

## 1. Delivered and verified

`scripts/audit_r8_reflections.py --shot <shot>` audits all 150 frames of a separate
`composite/ratio_r8_no_added_specular` experiment. This disables only the existing additive
GGX approximation (`--specular 0`). It keeps the original plate, leaves the rejected split and
catchlight replacement off, and uses the stored irradiance renders. The default compositor's
pixel calculations and arguments are unchanged. No source, key, normals, albedo or render
cache was regenerated or modified.

The experiment tests whether the *newly added* gloss is responsible for the lip and crown
complaints. Differences below are measured from the native 8-bit PNGs, before video encoding.
Means are equal-weight averages of per-frame region means; maxima are individual channels.

| Region | Mean removed display levels (0–255) | Largest channel change |
|---|---:|---:|
| Lips, excluding inner mouth | 0.00955 | 9 |
| Crown rectangle, including some forehead | 0.28577 | 46 |
| Iris candidates within eye openings | 0.07069 | 5 |
| Bright tooth candidates | 0.00619 | 1 |
| Bright sclera candidates | 0.03927 | 2 |

The lips and the original crown sheen remain visibly similar with added gloss off, including
the artist's frame 21. Disabling it is **not a demonstrated solution** and the default remains
1. The largest hair change also cautions against saying it does nothing everywhere. These
display-space differences cannot resolve sub-code-value changes or establish physical separation.

## 2. Eye tracking and the original catchlight: new diagnostic, failed recovery

The old claim that an *unclipped* reflection can therefore be subtracted to leave photographed
iris texture was unjustified. The recorded pixel is still an unresolved sum. Corrected that
claim in `eye_geometry.py`'s future log text, and its stale "replacement by default" description.
Historical cached geometry and old reports remain historical, not silently rewritten.

The new diagnostic registers each eye by its stored iris centre and mean radius. The median
radius is **12.34 px left, 12.21 px right** in this calculation. An interpolated 41x41 grid is
only a comparison coordinate system; it does not add captured resolution. Native eyelid
polygons are eroded before resampling; nearly shut eyes are excluded. Bright components
touching the aperture/iris boundary are rejected because they often belong to sclera.

There are sufficient visible grid pixels in **146 left-eye and 140 right-eye frames**. The
conservative candidate detector retains compact highlights in **59 and 108 frames** respectively.
All selected candidate locations have at least three nonadjacent, lower-radiance observations
outside the donor's detected highlight area. **That is not 100% recovery coverage.** It is
conditional on an incomplete, imperfect detector. The sequence review still shows occasional
lower-rim/sclera candidates, notably frames 67–69, and missed real reflections near lids.

To test recovery independently, withhold small discs of *known* non-candidate pixels at four
fixed iris-grid locations. Pick seven donor frames by agreement in the visible context, never
using the withheld RGB for ranking. Exclude the target and +/-3 adjacent frames. Use a median
of at least three visible donors outside their detected glints. Then compare the reconstruction
with the real withheld pixels, rather than congratulating it for making highlights dark.

| Withheld-pixel result | Left eye | Right eye |
|---|---:|---:|
| Supported test patches | 470 | 317 |
| Median relative RGB absolute error | 14.50% | 12.47% |
| 95th-percentile relative RGB absolute error | 50.10% | 64.59% |
| Median absolute display-channel error | 3.39/255 | 3.75/255 |
| 95th-percentile display-channel error | 11.98/255 | 11.83/255 |

**Rejected as a repair method.** Centre/scale registration and a temporal median are not
accurate enough to substitute iris pixels here. Gaze, pupil changes, eyelids, refraction,
blur, imperfect masks and registration remain confounds; this test does not attribute error
to any one of them. It also does not prove all temporal methods impossible. No recovered iris
pixels were written into a source plate or composite, and no replacement catchlight was added.

The code check perturbs all withheld target RGB values and verifies that donor selection and
prediction remain unchanged. Missing observations return zero supported pixels. This checks
the independence of the test; it does not make its failed reconstruction good.

## 3. Review artefacts and actual inspection

Under `music_videos/reporter_test/shots/reporter_a/final/`:

- `r8_hair_added_gloss_ablation.mp4`
- `r8_mouth_added_gloss_ablation.mp4`
- `r8_eyes_added_gloss_ablation.mp4`
- `r8_iris_tracking_evidence.mp4`

The first three compare round 4 on the **left** with added gloss disabled on the right, at
native pixels and 24 fps. These are diagnostic comparisons, not proposed improved delivery.
The iris video shows source samples and candidate overlays, enlarged for inspection, at the
shot's **60000/1001 fps**. It shows actual performance motion, not the stress-light tour.

All four videos decoded completely: **150 frames each, 600 total**. Every encoded frame was
visually inspected in ordered overview sheets (30 frames/page, five pages/video). Overview
pairs were resized to 492 px wide; JPG copies were used for three comparison sequences after
large PNGs failed the image-view tool. Native PNG pairs at frame 21 and the native enlarged
iris sequence at frames 12–17 were also inspected. **This was consecutive-frame inspection,
not real-time playback, and is not artist approval.**

Observed: the original lip band and crown sheen survive the ablation; eyes retain the original
softbox reflection; blink frames expose why iris/aperture tracking cannot be assumed exact.
Source-candidate overlays have both misses and false detections, so they must not become
automatic removal masks. The no-added-gloss comparison does not fix the underlying look.

Structured evidence under the shot's `logs/`:

- `r8_composite_manifest.json`: exact ablation settings and frame mapping.
- `r8_reflection_audit.json`: per-frame region differences, donor evidence and withheld tests.
- `r8_video_review.json`: complete decode, rates and sequence-sheet locations.
- `r8_code_checks.json`: withheld-RGB independence, missing-support and syntax checks.

Source comparison sheets, registered iris frames and native sequence sheets are in
`composite/r8_audit/`. Derived caches and videos follow the existing gitignore policy.

Reproduce in the pipeline venv:

```powershell
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/composite_ratio.py --shot music_videos/reporter_test/shots/reporter_a --name ratio_r8_no_added_specular --specular 0 --only-output
& dependencies/CorridorKey/.venv/Scripts/python.exe scripts/audit_r8_reflections.py --shot music_videos/reporter_test/shots/reporter_a
```

The audit intentionally refuses other shots because it retains the reporter crown ROI. It is
a reproducible experiment, not a generic production stage. The first compositor invocation in
this session omitted `--only-output`; its accidental update of `logs/composite_ratio.json` was
restored. The experiment's manifest is retained separately. The shared review-video log points
to the new iris diagnostic and each video has its own provenance sidecar.

## 4. Where the artist's three requests stand

1. **Existing eye catchlights:** still present. There is temporal variation worth exploiting,
   but the tested registration/recovery fails. Next useful work is reflection-aware iris
   correspondence and explicit uncertainty, alongside actual lid/source occlusion. Rendering
   another corneal pass cannot solve the missing original-reflection removal.
2. **Top-lit hair sheen:** still present. Added GGX is not its main explanation in frame 21.
   A global neutral-colour subtraction cannot be promoted from the rejected opening model.
   Hair needs material/strand-aware transport and a temporal fit that distinguishes pigment,
   shading and reflections while preserving strand contrast. None was validated this round.
3. **Lips reading too bright:** still present. Added GGX barely contributes on average. Do not
   retune global roughness to solve it, or re-enable the rejected separator because one
   lip/skin ratio improves while teeth fail. Lip material and the original gloss remain to
   be separated safely; relative brightness to surrounding skin alone is not ground truth
   under a different target-light direction.

These share *inherited source reflections*, but that does not establish one universal repair.
As external context, measured hair transport contains several lobes and fibre-orientation
effects, beyond a skin-like isotropic highlight: [Marschner et al., 2003](https://www.cs.cornell.edu/~srm/publications/SG03-hair-abstract.html).
Temporal priors can help intrinsic decomposition, but do not remove its ambiguity:
[Live Intrinsic Video](https://vcai.mpi-inf.mpg.de/projects/LiveIntrinsicVideo/).
Neither source validates a solution on this footage. They support investigating material and
temporal models, not promising that a new filter will achieve full photorealism.

## 5. Settled

Keep round-4-equivalent defaults. Rejected split and catchlight replacement remain off.
No on-set reference requirement, no key-threshold change, no persistent Cycles data, no
generative performer replacement. Preserve historical evidence. Syntax and focused code checks
pass; the temporal **image-quality** experiment fails. Those are different conclusions.

## 6. Starter prompt for round 9

> Read round8.md, round7.md and round6.md's rejection addendum. Round 8 did not fix the three
> original asks. Do not repeat the added-gloss ablation or use centre/scale temporal medians as
> a validated iris repair. If pursuing eyes, use the withheld-pixel test as an independent
> control, improve reflection-aware correspondence and mask reliability, and handle original
> reflection removal plus lid/source occlusion together. Candidate donor coverage is not clean
> texture recovery. If pursuing hair/lips, start a material-aware, temporally constrained
> experiment with original-plate anatomical controls, not morphological-opening subtraction.
> Produce complete comparison videos and inspect the whole sequence before any promotion.
> Keep experiments off by default and state plainly if the original three issues remain.
