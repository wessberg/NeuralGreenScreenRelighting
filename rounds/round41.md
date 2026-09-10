# Round 41 handoff — the tour's bulb reflects in her corneas, on top of the accepted removal

**Date:** 2026-09-10 / 11. Starts from round 40. **The artist accepted the softbox removal**
(*"I'm pretty satisfied with our removal of the softbox reflection from the eyes now"*) — that is
round 40's verdict, `r40_floorB` is the removal baseline — and moved the work on: *"make sure the
lightbulb reflects in her eyes correctly"*, on the bulb-tour scene (`stress_proxy.py`,
`reporter_a`), and asked whether the reflection can be driven by Blender so it shows the actual
objects casting the light, their hue, and secondary reflections. It can, and that is what was
built: nothing here is a stock shape. Round 40's ellipse-fit plan is superseded; see §6.
**Shown to the artist:** `reporter_a/final/r41_tour_compare.mp4` (before = the tour with the
capture room's softbox relit, LEFT; after = removal + corneal reflection, RIGHT),
`final/r41_tour_eyes.mp4` (four rows at eye scale: source / before / removal only / final, every
frame), `final/r41_tour_panel.mp4` (face scale). **Verdict (2026-09-11):** *"looks decent enough. Maybe slightly
unrealistic? Maybe the eye didn't become 'wet' enough?"* — accepted as a step, not as done; the
wet look is round 42's target (§6 items 2–3: the reflection stops at the cornea, the sclera and
the tear film along the lids do not respond to the new light, and the capture room's own
scleral glint is still in the photograph).

---

## 0. What was built

The chain, on `reporter_a` (source 700–849 of the plate, the 150 frames the tour and the
ratio composite live on):

1. **`scripts/import_overlap_eyes.py`** — copies the ACCEPTED `r40_floorB` frames and the tracks
   they were built with (`limbus_track_tv.json`, `eyelid_track_wtv40.json`) from `reporter_full`
   onto `reporter_a`, renumbered by the offset both `shot.json`s record (700), **gated per frame
   on the keyed fg EXR's sha256 matching the run's own record** (150/150 matched: the key ran
   twice with the same thresholds and wrote byte-identical files). Copies, never recomputes: a
   fresh 150-frame eye pass pools its canonical iris and lid curvature over a different clip
   length and would not be the frames he judged. Writes `composite/r40_floorB_import/` and
   `relight/*_import.json`.
2. **`stress_proxy.py --corneal`** — the round 34 corneal construction lit by THE TOUR: a
   dielectric corneal CAP (Principled, black base, roughness 0.02, IOR 1.376; truncated at the
   limbus, sin θ = 5.85/7.8) at each eye, the plate camera, plate resolution, linear EXR, no
   denoiser (a bulb images as a sub-pixel spot and a denoiser calls that noise), bulb and fill
   hidden from camera rays but seen by reflection rays, proxy ABSENT (round 34: no eye sockets,
   it buries the cornea; `--occluder` puts it back). Same `EXPOSURE`, same `station_at`, same
   hue/pulse/blackout as `--ratio`, in the same file, so the reflection and the irradiance
   field cannot be lit differently. ~0.7 s/frame. Manifest: bulb/fill world position, colour,
   strength, per-eye sphere centre, radius, gaze, anchor pixel, and whether the eye rendered.
3. **The sphere is anchored on the IRIS CENTRE from the limbus track, behind it along the
   measured gaze** (`corneal_anchor.anchor_limbus`): the tracked pixel is the centre of the iris
   disc, which sits √(R² − r²) = 5.16 mm in front of the sphere centre; `anchor_centre` had put
   the corneal APEX on the ray (2.64 mm in front of the iris centre, 1.7 px sideways at 18° off
   axis). Gaze from `eye_geometry.json` (head pose + iris displacement); `--gaze camera` is the
   alternative. Which one is right was decided by the photograph (§1).
4. **`composite_ratio.py --eye-removal <run> --corneal-dir <pass>`** — the eye pass's pixels
   replace the plate wherever its weight is non-zero (8-bit sRGB → linear; exact fg elsewhere),
   the ratio multiplies THAT, and the corneal pass is ADDED afterwards through a matte of the
   open cornea — the fitted upper and lower lids (plus round 40's `lower_iris` where it
   exists) AND the limbus disc, feathered ~1 px — scaled by **the same `anchor × exposure` the
   plate is multiplied by**, so render and plate share units with no dial. Without a lid fit
   (42 of 300 eyes: around the two blinks and, for the right eye, 135–149) it falls back to
   the mediapipe eyelid polygon exactly as the eye pass does; with neither, nothing. Closed
   eyes (no limbus fit, 28) get nothing. `--corneal-blur` softens the pass to the plate (§2).
   Old round-5 `--catchlight` path untouched and still off.
5. **`scripts/validate_tour_catchlight.py`** — two independent numpy checks (§1, §3).
6. **`scripts/measure_glint_psf.py`** — the plate's and the render's sharpness, on the glint
   itself (§2).

## 1. The photograph grades the sphere (`--plate-glint`)

The capture room's softbox is static, so its glint, measured on every OPEN eye of the untouched
source (luminance-weighted centroid of the brightest blob inside the limbus; lid opening ≥ 1.4 r
at the centre column), must be explained by ONE distant direction reflected in the placed
sphere. The direction is fitted (two angles, global search); the residual is placement error.
212 open eyes:

| sphere placement | rms px | p95 px | max px | bias L / R (x, y) px | residual step p95 |
|---|---:|---:|---:|---|---:|
| iris-centre anchor, gaze toward the camera | 1.87 | 2.80 | 7.22 | (+1.24, −0.17) / (−1.35, +0.17) | 0.36 |
| **iris-centre anchor, `eye_geometry` gaze** | **1.31** | **1.71** | 8.66 | (+0.25, +0.04) / (−0.27, −0.03) | 0.45 |

The camera-axis model leaves a 1.3 px lateral bias of OPPOSITE sign on each eye — the measured
gaze carries real information about where the corneal sphere sits, even with the 18° scatter
round 34 recorded. Smoothing the gaze (5-frame median) changed nothing (1.30). The fitted
softbox direction is (−0.47, −0.81, +0.36), 36° off axis, up and to camera-left, where the
plate's glint is. `eye_geometry.py`'s own `plate_glint` centres were tried first and are 3–5 px
noisy; at a blink the visible part of the glint moves with the lid, which is why the open-eye
gate is not optional. Frame-to-frame residual step p95 0.45 px: the placement does not flicker.

## 2. Sharpness, measured on the glint itself (`measure_glint_psf.py`)

`b + A · erf-box(W, H, σ)` fitted to the softbox glint on 210 open eyes of the plate, and to the
tour's bulb spot on the render (the disc size fixed from the geometry: r_img = (R/2)·r_bulb/d).

| | σ median px | p5 | p95 | n |
|---|---:|---:|---:|---:|
| plate softbox glint (box 3.3 × 4.0 px) | 0.473 | 0.100 | 0.603 | 210 |
| render, far bulb (spot r < 0.25 px) | 0.318 | 0.271 | 0.485 | 22 |
| render, near bulb (r > 0.5 px, disc fixed) | 0.405 | 0.389 | 0.417 | 35 |

`extra = √(0.473² − 0.318²) = 0.35 px` → `--corneal-blur 0.35`. Small, but it is the honest
number: the render's only spread is the 1.5 px Blackman-Harris pixel filter.

## 3. The tour grades the render (`--render`)

Per rendered eye, the bulb reflected in the same sphere by `corneal_anchor.reflect_point`
(Alhazen by fixed point), projected, against the rendered peak — and three deliberately wrong
conventions. 185 eyes with the bulb in the cap's field:

| convention | offset median px | p95 | max |
|---|---:|---:|---:|
| **reflection** | **0.20** | **0.41** | 21.1 |
| bulb mirrored in X | 1.27 | 18.5 | 22.3 |
| bulb mirrored in Z about the eye | 4.20 | 16.6 | 19.9 |
| surface normal instead of reflection | 5.47 | 6.08 | 25.7 |

Chromaticity of the rendered peak vs the bulb's colour: 6e-5 median (Fresnel is achromatic; the
0.25 max is frames where the fill's wash is the peak). Bulb behind her: 2.5 % of the front
energy in the eye (the fill, which is then in front). The three > 1 px "reflection" outliers
are 80/82/84 right at *camera right*, where the bulb reflects at the cap's edge and the peak
found is the fill. Sphere recomputed by the validator vs the manifest's: 3e-16 m.

## 4. What is in the eye, per station (`logs/r41_corneal_content.json`, plate units)

| station | bulb peak | bulb px | other max | other rho | other px > 0.05 |
|---|---:|---:|---:|---:|---:|
| front far (2.4 m) | 1.60 | 2 | 0.13 | 0.23 | 1 |
| front close (0.5 m) | 1.14 | 3 | 0.14 | 0.28 | 3 |
| below chin | 1.92 | 3 | 0.15 | 0.66 | 3 |
| above head | 1.61 | 3 | 0.22 | 0.58 | 4 |
| camera left | 2.92 | 2 | **0.52** | **0.83** | 18 |
| camera right | 1.61 | 14 | **0.51** | **0.88** | 15 |
| behind rim | 0.44 | 55 | 0.07 | 0.41 | 2 |
| KEY OFF | 0 | — | 0 | — | 0 |
| high three-quarter | 0.97 | 2 | 0.12 | 0.41 | 1 |

* **The bulb is a 2–3 px spot that clips to white in the 8-bit review**, with its hue in the
  falloff pixels. That is physics: a 4 cm bulb bright enough to key a face reflects at ~500× the
  face's radiance and a camera clips it too; the HDR master keeps the values (peak 1.1–2.9
  linear). If he wants to SEE the hue at review, that is a review tone-map decision, not a
  render change — do not dim the bulb to make it pretty.
* **"Behind rim" and the key-off → three-quarter transition (frame 127) show a soft grey
  square in each pupil: the 60 cm FILL panel**, in front of her at those stations, imaging at
  (R/2)·0.6/0.6 ≈ 4 mm → ~8 px. Physically that is what a soft source does — the plate's own
  softbox is the same object — and it is a secondary reflection of the kind he asked about.
* **At *camera left* / *camera right* (frames 52–85, 44 eyes) the fill lands at rho 0.83–0.88,
  up to 0.5 linear: a pink crescent cut off at the limbus.** The panel is 60 cm at ~0.55 m to
  the side and slightly behind the eye plane, so it reflects at the very edge of the cornea;
  in a real head it continues onto the wet sclera and is partly hidden by the nose bridge
  (nasal side) or the temple and lashes (temporal side). The cap truncation makes a hard-edged
  crescent of it. Left in, on purpose, and it is the first thing to ask him about (§6).

## 5. Facts worth not re-deriving

* 272 of 300 eyes carry the reflection; the 28 without are the closed eyes of two blinks (no
  limbus fit). 42 use the mediapipe polygon matte (no lid fit); stepped through at 7× on
  3–4, 12–23, 127, 134–149: the spot sits on visible iris, never on a lid.
* `composite_ratio.py` anchor unchanged by the substitution (10.2250 before and after);
  removal px per frame equal `removal.json`'s `changed_pixels` (594 at frame 0).
* **Cycles on the GPU is not bit-reproducible run to run**: two renders of the pass differ by
  up to 0.0127 linear on 121 of 150 frames (noise level; every geometry record identical).
  Do not gate anything on an EXR hash of a render.
* `eye_geometry.json`'s `plate_glint` centres are too noisy (3–5 px) to grade a sphere; the
  validator measures its own.
* Anchor apex-vs-iris-centre difference: 1.38 mm along the gaze, as computed.
* Timing: corneal pass ~2 min / 150 frames; composite ~2.5 min; the validator's global
  direction search ~1 min per model.

## 6. Still open — the round 42 list

1. **His verdict** on `r41_tour_compare.mp4` / `r41_tour_eyes.mp4` / `r41_tour_panel.mp4`, and
   specifically on the two fill behaviours in §4: the grey square (behind rim, 127) and the
   crescent (52–85). Whatever he raises outranks the rest.
2. **The reflection stops at the limbus; the eye does not.** The sclera is wet and reflects
   too (round 38 logged the capture room's second glint on the white of the eye). A second,
   larger glossy sphere (≈12 mm, centre ~5.6 mm behind the corneal centre) with a rougher,
   dimmer lobe, matted by the lids only, would carry the fill crescent past the limbus the
   way a real eye does and remove the hard edge. New construction — fresh round.
3. **No occlusion by her own head.** The proxy has no sockets (buries the cornea, round 34), so
   a source across the nose bridge reflects when it should not (camera left/right, the nasal
   eye). A cheap geometric socket — a cone of allowed directions per eye, wider temporally
   than nasally — would be the honest next step; measure it against the plate first (which
   softbox directions ever appear in which eye).
4. **Review tone-mapping** for the clipped bulb, if he wants its hue visible at review; the
   master keeps it.
5. For his own footage the chain needs `eye_geometry.py`, `limbus_fit.py`, `limbus_trust.py`,
   `eyelid_fit.py --lower-iris-edge`, the eye pass, then `--corneal` and the two composite
   flags — no `import_overlap_eyes.py`, which exists only because the removal and the tour
   were built on two shots of one plate.

## 7. Reproduce

```powershell
$P = "dependencies/CorridorKey/.venv/Scripts/python.exe"
$B = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
$S = "music_videos/reporter_test/shots/reporter_a"
& $P -B scripts/import_overlap_eyes.py --shot $S --from music_videos/reporter_test/shots/reporter_full --run r40_floorB --tracks limbus_track_tv.json,eyelid_track_wtv40.json
& $B -b --factory-startup --python scripts/stress_proxy.py -- --shot $S --corneal --length 0 --samples 256 --name r41_corneal
& $P -B scripts/validate_tour_catchlight.py --shot $S --plate-glint --render composite/r41_corneal --gaze geometry --out r41_corneal_validation.json
& $P -B scripts/measure_glint_psf.py --shot $S --render composite/r41_corneal          # -> extra sigma 0.35
& $P -B scripts/composite_ratio.py --shot $S --name r41_tour_removal --eye-removal r40_floorB_import          # control
& $P -B scripts/composite_ratio.py --shot $S --name r41_tour --eye-removal r40_floorB_import --corneal-dir r41_corneal --corneal-blur 0.35
& $P -B scripts/review_r35_eyes.py --shot $S --dirs "r40_floorB_import/source,ratio/.,r41_tour_removal/.,r41_tour/." --labels "SOURCE|TOUR (softbox relit, before)|TOUR + R40 removal|TOUR + removal + corneal reflection" --track limbus_track_tv_import.json --name r41_tour_eyes --per-sheet 30
& $P -B scripts/review_r37_panel.py --shot $S --name r41_tour_panel --dirs "ratio/.,r41_tour/." --labels "TOUR before (softbox relit)|TOUR + removal + corneal reflection"
& $P -B scripts/review_video.py --shot $S --name r41_tour --compare ratio --label "R41: removal + corneal reflection" --compare-label "before: softbox relit" --out r41_tour_compare.mp4
```

## 8. Starter prompt for round 42

> Read round41.md §4 and §6 first. Round 41 put the bulb tour's own reflection into her corneas
> on top of the accepted removal: `stress_proxy.py --corneal` (round 34's dielectric cap, lit by
> the tour, anchored on the fitted iris centre behind the measured gaze — chosen because the
> plate's own softbox glint grades it at 1.31 px rms vs 1.87 for a camera-axis gaze),
> `composite_ratio.py --eye-removal --corneal-dir --corneal-blur 0.35` (the pass added through
> the fitted lids and limbus, in the ratio's own units, softened to the plate's measured 0.47 px
> PSF). Validated: 0.20 px median against an independent reflection model, wrong conventions at
> 1.3–5.5 px. Run `r41_tour`; videos `r41_tour_compare.mp4`, `r41_tour_eyes.mp4`,
> `r41_tour_panel.mp4`. **Ask for / record his verdict first**, especially on the fill panel's
> reflection: a soft grey square at "behind rim" and frame 127, and a pink crescent cut at the
> limbus at camera left/right (frames 52–85). Whatever he raises outranks this list.
>
> Then the two known limits, both new construction: the reflection must continue past the
> limbus onto the wet sclera (a second, larger, rougher glossy sphere matted by the lids only),
> and a geometric socket so a source across the nose bridge does not reflect in the nasal eye
> (measure which softbox directions the plate itself shows in which eye before building it).
> Keep: GENERIC tool, GPU first, the plate's pixels outside the eye pass's weight exact, p95 and
> max, every changed eye looked at against the previous run before a video. Do not dim the bulb
> to make its clipped spot prettier; that is a review tone-map decision. Cycles renders are not
> bit-reproducible (0.013 linear); never gate on a render hash.
