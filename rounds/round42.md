# Round 42 handoff — the wet eye: a scleral globe in a head socket, the old room's glints off the white

**Date:** 2026-09-11. Starts from round 41 (`r41_tour`, verdict *"looks decent enough. Maybe
slightly unrealistic? Maybe the eye didn't become 'wet' enough?"*). This round builds the wet
eye as ONE construction on the existing geometry and removes the capture room's reflections
from the white of her eye. **Shown to the artist:** `reporter_a/final/r42_wet_compare.mp4`
(round 41 LEFT, round 42 RIGHT), `final/r42_wet_eyes.mp4` (source / round 41 / round 42, every
frame at eye scale), `final/r42_wet_panel.mp4` (face scale). **Verdict (2026-09-11): *"Yeah, it's a huge improvement! Looks super good."* ACCEPTED. He asked why the bulb is a tiny speck when the softbox was enormous: a catchlight is a mirror image, its size is the source's angular size times half the corneal radius; a 4 cm bulb at 0.5–2.4 m subtends 1–5° and images sub-pixel (bright, not big), the ~1 m softbox subtended ~35° and imaged 3×4 px. A large soft reflection in his scene needs a large source in the scene, not a render setting.**

---

## 0. What was built

1. **`audit_r42_wet.py`** — measured first (§1): where the capture room's light shows on the
   parts of her eye that are not the iris, per open eye, and what a scleral globe and a lower-lid
   tear meniscus would predict for it.
2. **`stress_proxy.py --corneal --globe`** — the eye is now ONE surface of revolution about the
   gaze (`eye_profile` / `revolve`): the 7.8 mm corneal sphere, a 1 mm concave limbal fillet
   tangent to both spheres (`--fillet-mm`), and the 12 mm scleral globe centred 5.6 mm behind
   the corneal centre (`--globe-mm`, `--globe-back-mm`; the spheres meet at 6.28 mm = 1.07 iris
   radii). Two materials on one mesh: the round 34 cornea (black Principled, roughness 0.02,
   IOR 1.376) and the same dielectric with `--sclera-roughness` 0.08 for the tear film over the
   conjunctiva. Both radii and the roughness are **ASSUMED anatomy** (§1 says why nothing fits
   the roughness). The manifest records the globe per eye.
3. **`--socket`** — a black diffuse CUP about the globe centre, 24 mm, attached to the HEAD
   (mediapipe `xform` from `face_landmarks.npz` through the convention `eye_geometry.py`
   verified; cross-checked here 10–15° from the verified gaze on the four test frames, never
   flipped), open only inside the visual-field limits (nasal 60°, temporal 100°, up 60°, down
   75°, cos²-interpolated by azimuth; `--socket-*-deg`), hidden from camera rays. Without it
   the globe sees BACKWARDS: at the front-far station the fill panel is 1.44 m *behind* her,
   and the globe painted its reflection as a bright arc on the lower-temporal sclera of both
   eyes (§3). It is round 41 §6 item 3, built because it became necessary rather than optional.
4. **`sclera_glint_removal.py`** (GPU) — removes the old room's glints from the white of her
   eye on top of the accepted iris run (§2), in the eye pass's own layout so
   `composite_ratio.py --eye-removal` takes both at once.
5. **`composite_ratio.py --corneal-matte lids`** — the corneal pass through the fitted lids
   ONLY (no limbus gate) when the pass carries the globe.
6. **`validate_tour_catchlight.py --globe`** — the bulb reflected in the globe by the same
   Alhazen solve, scored against the NEAREST rendered local maximum outside the junction (the
   globe shows the fill's broad lobe and the bulb at once, and at some stations the lobe is the
   brighter, so "is the brightest peak where the model says" is the wrong question), with
   wrong-convention controls plus a fifth, `cap_not_globe` (§4). The cap check reads the cap's
   half-angle from the manifest now that the cap extends to the junction.

## 1. Measured first (`logs/r42_wet_audit.json`, `composite/r42_wet_audit/sheet.png`)

On every open eye of the untouched source (lid fit present, opening ≥ 1.4 r at the centre):

| | left (109 open eyes) | right (103) |
|---|---|---|
| blobs on the BULK sclera (≥ 2.5 px from both lids) | 22 eyes, rho 1.55, −156°, 100 % nasal | **142 blobs / 100 eyes**, rho 1.65, **−151°**, 6.9 px above the lower lid, 68 % temporal |
| blobs on the lower-lid margin (< 2.5 px above it) | 36 eyes, rho 1.49, −146° | 1 |
| corner (tear-lake) blobs | 6 | **95 eyes**, rho 1.77, −13°, nasal |
| lower-lid band 1 px above the parabola vs the sclera 4.5 px up | excess median 0.09, p95 0.17 | 0.07, p95 0.18 |
| softbox reflected in a 12 mm globe (round 41's fitted direction) | rho 0.63, max 0.70 — **under the cornea on 100 %** | rho 0.36, max 0.58 — 100 % |
| tear-meniscus tube along the lower lid, predicted glint column vs the band's brightest column | error 9.4 px median | 19.3 px |

Three consequences, all of which shaped the build:

* **The capture room's softbox never reflects in the visible sclera** when she looks at the
  camera: on a globe the reflection point of a source 36° off axis sits 0.36–0.70 iris radii
  from the centre, i.e. under the cornea. So the plate offers NOTHING to fit the sclera's
  roughness or level to, and `--sclera-roughness` stays an assumption.
* **The blob on the white of her right eye is not the softbox in a globe** — it sits at
  −151°, lower-temporal, the same image angle as the left eye's, 7 px above the lower lid; a
  reflection at that place needs a source ~40° down-left of the camera, and the cornea shows
  no matching second glint at rho 0.45 (looked at 16×, gamma-lifted). It is a real reflection
  of the OLD room on a surface this model does not have (unresolved — logged, not invented).
  It is removed (§2) and the rendered globe now answers the NEW scene there.
* **There is no tear line to build.** The lower-lid band is at most 0.09 display above the
  sclera beside it and its brightest column is at the corners (±1.2–1.6 r), which is the tear
  lake, not a meniscus; a tube along the lid predicts glint columns 9–19 px from where the
  band is brightest. Round 41's starter prompt asked for the lid margin to respond to the new
  light; the plate shows no highlight to replace, so a rendered line would be an invention.
  Not built.

## 2. The old room's glints off the white (`composite/r42_wet_removal/`)

The audit's own blob test, applied to the accepted iris run's frames (`r40_floorB_import`):
bulk and margin blobs inside the fitted opening and outside 1.10 r; corner blobs LEFT ALONE
(a wet caruncle glints, and nothing rendered replaces it yet; `--corners` removes them). The
footprint is held in time — hot now, or hot in the previous/next frame registered on the
limbus centre and still warm now — and filled by normalised convolution of the sclera around
it (σ 2.5 px, the footprint dilated 2 px excluded), through a 1 px feather. 12.6 s on the GPU.

| | left | right |
|---|---:|---:|
| eyes changed / seen | 64 / 137 | 111 / 135 |
| footprint px median / max | 1 / 6 | 3 / 6 |
| level before → after (display luma, median) | 0.81 → 0.63 | 0.82 → 0.55 |
| filled level frame step p95 / max | 0.054 / 0.140 | 0.020 / 0.039 |
| control ring (untouched sclera 3–5 px out) step p95 / max | 0.052 / 0.133 | 0.026 / 0.094 |
| corner px left in place | 17 | 220 |

The filled region's frame-to-frame step is the same as the untouched sclera's beside it on
both eyes: the fill moves with the eye, not on its own.

## 3. The globe saw backwards; the socket

The first full globe pass (`composite/r42_globe`) put a bright arc (0.7 linear) on the
lower-temporal sclera of both eyes at the front-far station, inside the lid matte. Traced
pixel by pixel (reflected direction against the manifest): within 10–23° of the FILL, which
that station parks 1.44 m behind her. A globe with no head around it reflects the room behind
the head. Round 34 measured that the proxy cannot be the occluder (no sockets, buries the
cornea); the cup is the cheap geometric socket round 41 proposed. Two things it settled:

* **Cup radius 14 mm was wrong.** A ray leaving the 12 mm globe at its edge exits a 14 mm cup
  well inside its own direction angle, and the behind-rim bulb (134° from head-forward)
  still glinted on the temporal edge of her right eye. At 24 mm (about the orbital rim) exit
  angle and direction agree and it is gone.
* **Nasal cut 60° behaves like a nose.** At camera left the right eye's fill crescent (nasal
  side) is cut and the left eye's (temporal) is whole; at camera right the mirror. The bulb at
  camera right, 69° nasal for her right eye, no longer reflects in it. Real heads do that.

## 4. Validation of the final pass (`logs/r42_eye_pass_validation.json`, `r42_glint_psf.json`)

Cap check (`--render`, as round 41): 271 eyes rendered, 186 with the bulb inside the extended
cap. Reflection offset **0.23 px median**; p95 20.5 / max 22.0 are the camera-right right eyes
(71–85) where the bulb reflects at the cap's very edge, the socket now blocks it, and the peak
the disc window finds is the fill — round 41's outlier class, larger because the cap extends to
the junction. Front far / front close / above head / camera left: median 0.18–0.26, max ≤ 0.43.

Globe check (`--globe`), 136 eyes whose predicted globe reflection lies outside the junction:

| | n | offset median px | p95 | max |
|---|---:|---:|---:|---:|
| socket OPEN by the renderer's own rule, peak present | 25 (all 25 open eyes) | **0.04** | 0.15 | 0.21 |
| control: bulb mirrored in X | 25 | 1.22 | 4.11 | 6.48 |
| control: bulb mirrored in Z | 25 | 20.0 | 21.1 | 21.1 |
| control: normal instead of reflection | 25 | 7.67 | 8.04 | 8.37 |
| control: reflected in the CAP, not the globe | 25 | 6.03 | 6.35 | 6.36 |
| socket BLOCKED by the rule | 111 | — | — | — |

Open eyes with no rendered peak: 0. Blocked eyes that nonetheless render a peak within 1.5 px of
the unblocked prediction: **14 of 111** — her left eye at camera left (57–68), one right eye at
72, one at 85: the bulb 60–66° nasal, i.e. within a few degrees of the 60° nasal cut, where the
validator's rule (from the surface point) and the mesh cup (cut by angle from the globe centre)
disagree by the parallax of a 12 mm globe in a 24 mm cup. The X-mirror control is weak here
(1.2 px) because the 25 scorable eyes are mostly the below-chin station, where the bulb sits
near x = 0; the Z-mirror, normal and cap controls carry the discrimination.

Globe recomputed by the validator vs the manifest's centre: 3e-16 m. Sharpness on the glint
(`measure_glint_psf.py`): extra sigma to reach the plate **0.30 px** (round 41: 0.35; the same
`--corneal-blur 0.35` was used, 0.05 px over).

Scleral removal + composite: anchor 10.2250 unchanged; `r42_wet` = `r42_wet_removal` under the
ratio + `r42_eye_pass` through the fitted lids only (mediapipe polygon on the eyes with no lid
fit, nothing where no limbus fit). `r42_wet_nofillet` is the same composite from the pass
without the fillet (`r42_wet_pass`), kept for comparison; `r42_globe` is the first globe pass
with no socket (§3).

## 5. Facts worth not re-deriving

* **Two spheres meeting at the limbus reflect a source 60–80° off axis TWICE**, 5 px apart:
  once on the cornea (normal 35° from the view axis → 4.5 mm from the axis) and once on the
  globe (same normal → 6.9 mm). A 1 mm concave fillet between them sweeps the normal BACK
  (53° → 31°) so it cannot bridge the two; a real limbal sulcus has the same property, and
  macro photographs of a side-lit eye do show a corneal glint at the limbus and a second on
  the sclera. The globe's spot is ~3× brighter than the cap's for the same source because its
  image is (12/7.8)² larger and the bulb's image is sub-pixel either way. Kept; his eye
  decides.
* The globe's fill lobe at the side stations reaches 0.5–1.0 linear at grazing Fresnel and
  clips to white in the 8-bit review; a 60 cm panel 55 cm to the side IS that bright in a wet
  eye. The master keeps the values; review tone-mapping is still round 41 §6 item 4.
* `local_maxima_outside`: scoring against the brightest blob outside the junction gave 34 px
  "errors" at camera right that were the fill's lobe out-shining a 0.10 bulb glint that WAS
  where the model said. Nearest-peak scoring per convention keeps the controls honest (wrong
  conventions still land 6–38 px off) without asking the bulb to be the brightest thing.
* **The 60 cm fill panel at 0.6 m subtends ~53°**, so its image straddles the junction: the
  cornea shows one part (the grey square in the pupil, round 41 §4) and the globe another, a
  second grey square beside the first at the limbus (127, behind rim, right eye). At camera
  left/right the globe's lobe of it is a flat pink/white wedge across the temporal sclera that
  clips in the review. Both are the scene as built, not a rule; both will read odd and are the
  first things to ask him about (§6).
* The socket's borderline: the validator's angle rule and the mesh cup disagree within ~5° of a
  cut (14 eyes, §4). Not a bug worth geometry; the cut angles are assumptions anyway.
* `--only-output` takes SHOT-LOCAL frame numbers (the proxy mesh names, 0–149), not source
  frame numbers.
* `$TMP` in this harness is the Windows temp dir, not the session scratchpad.

## 6. Still open — the round 43 list

1. **His verdict** on `r42_wet_compare.mp4` / `r42_wet_eyes.mp4` / `r42_wet_panel.mp4`.
   Specifically: the double glint at the limbus at camera left/right (52–85), the clipped
   white fill lobe on the sclera at the same stations, and whether the corner tear-lake glint
   of the OLD room (left in place, 95 right eyes) now reads wrong beside a new-scene sclera.
2. The lower-temporal blob of the old room on her right eye (§1) has no surface in this model.
   If it matters for other footage, find what reflects there (a second source down-left of the
   camera would also glint on the cornea at rho 0.45; it does not).
3. Sclera roughness and the socket's four angles are assumed. A plate with a source far
   enough off axis to reflect in the visible sclera would fit the first; a head turn across a
   known light would fit the second.
4. Review tone-mapping for the clipped bulb and fill lobe.
5. On his own footage the chain is `eye_geometry.py`, `limbus_fit.py`, `limbus_trust.py`,
   `eyelid_fit.py --lower-iris-edge`, the iris pass, `sclera_glint_removal.py`, then
   `stress_proxy.py --corneal --globe --socket` and `composite_ratio.py --eye-removal
   --corneal-dir --corneal-blur 0.35 --corneal-matte lids`.

## 7. Reproduce

```powershell
$P = "dependencies/CorridorKey/.venv/Scripts/python.exe"
$B = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
$S = "music_videos/reporter_test/shots/reporter_a"
& $P -B scripts/audit_r42_wet.py --shot $S --softbox r41_corneal_validation.json
& $P -B scripts/sclera_glint_removal.py --shot $S --run r40_floorB_import --name r42_wet_removal
& $B -b --factory-startup --python scripts/stress_proxy.py -- --shot $S --corneal --globe --socket --length 0 --samples 256 --name r42_eye_pass
& $P -B scripts/validate_tour_catchlight.py --shot $S --render composite/r42_eye_pass --gaze geometry --globe --out r42_eye_pass_validation.json
& $P -B scripts/measure_glint_psf.py --shot $S --render composite/r42_eye_pass --out r42_glint_psf.json
& $P -B scripts/composite_ratio.py --shot $S --name r42_wet --eye-removal r42_wet_removal --corneal-dir r42_eye_pass --corneal-blur 0.35 --corneal-matte lids
& $P -B scripts/review_r35_eyes.py --shot $S --dirs "r40_floorB_import/source,r41_tour/.,r42_wet/." --labels "SOURCE|R41: cap only|R42: globe + socket, white cleaned" --track limbus_track_tv_import.json --name r42_wet_eyes --per-sheet 30
& $P -B scripts/review_r37_panel.py --shot $S --name r42_wet_panel --dirs "r41_tour/.,r42_wet/." --labels "R41 cap only|R42 wet eye"
& $P -B scripts/review_video.py --shot $S --name r42_wet --compare r41_tour --label "R42: wet eye" --compare-label "R41: cap only" --out r42_wet_compare.mp4
```

## 8. Starter prompt for round 43

> **The wet eye is ACCEPTED** (*"a huge improvement! Looks super good"*), so round 43 starts from `r42_wet` as the eye baseline. Read round42.md §1, §3, §4,
> §5 and §6 first. Round 42 made the eye ONE surface (cornea + 1 mm limbal fillet + 12 mm
> globe, `stress_proxy.py --corneal --globe --socket`), put it in a head-attached socket cup so
> it stops reflecting the room behind her head, removed the capture room's glints from the
> white of her eye on top of the accepted iris run (`sclera_glint_removal.py`), and composited
> the pass through the fitted lids only (`composite_ratio.py --corneal-matte lids`). Videos:
> `reporter_a/final/r42_wet_compare.mp4` (R41 left, R42 right), `r42_wet_eyes.mp4`,
> `r42_wet_panel.mp4`. Three things the construction produces that are physics, not rules, and
> that he may reject on sight: the DOUBLE glint at the limbus for a source 60–80° off axis
> (52–85); the fill panel's flat clipped lobe across the temporal sclera at camera left/right;
> the second grey square beside the corneal one at 127. If he rejects them, the honest levers
> are (a) review tone-mapping (the master keeps the values), (b) the ASSUMED sclera roughness
> (0.08; nothing in this plate fits it — the softbox reflects in the globe under the cornea on
> every open eye), (c) a smaller or farther fill in the TOUR, which is a demo rig, not her
> scene. Do not dim the bulb; do not hide a defect with a mask.
>
> Still not built and measured why: a tear line along the lower lid (the plate has none to
> replace, §1); the old room's lower-temporal blob on the right eye has no surface in this
> model (§1) and is removed, not explained; the corner tear-lake glint of the old room is left
> in place (95 right eyes) because nothing rendered replaces it — if it now reads wrong beside
> the new-scene sclera, `--corners` removes it and the caruncle needs its own reflector.
>
> Keep: GENERIC tool, GPU first, plate pixels outside the eye passes' weight exact, p95 and
> max, `--corneal` and `--ratio` lit by the same code, wrong-convention controls on every
> geometric claim, every changed eye against the previous run at eye scale before a video.
> On his own footage the chain is round42.md §6 item 5.
