# Round 34 handoff — the eye is GENERATED ONCE, and the tracker is the blocker

**Date:** 2026-09-09. Starts from round 33. Production defaults unchanged.
**Delivered:** the reason rounds 32–34 all looked wrong, three released 2026 models
evaluated head to head, a construction that makes the eye temporally stable *by
construction*, and the measurement that identifies what now blocks it.
**Not delivered:** artist acceptance. Photorealism is not met.

**The artist's verdict, in his words, and it is the direction for round 35:**

> the good news is that the new iris generation approach here seems to be the right one

and, on why it is not yet acceptable:

> your new iris drifts away from the actors own iris several times, giving her a double
> iris … our synthesized pupil seems to be not black enough compared to the source …
> It feels mostly like a mask that is "delayed" or slow to react … it just isn't
> tracking exactly

He is right, it is measured in §5, and it is upstream of everything.

---

## 1. The one thing to take from this round

**The eye is now built ONCE, in the eye's own registered frame, and rendered into all
150 frames.** Not detected per frame, not filled per frame, not generated per frame.
`eye_soft_removal.py --fill-iris` replaces the whole iris disc from a single canonical
image; the only thing that varies per frame is a scalar illumination gain. Frame-to-frame
crawl of the iris content is not *reduced* by this, it is **structurally impossible**.

That is the answer to the failure that killed rounds 31, 32, 33 and this round's own
first three attempts, and it is the same answer every time: **anything refit per frame
flickers.** Round 31 asked DiffusionRenderer to redraw the eye each frame. Round 32 refit
every term of its fill each frame. Round 33 fixed the fill and left the *mask's* threshold
per frame. Round 34's ProPainter pass composited the model's per-frame output directly —
and the artist caught the crawling pupil before any metric here did.

## 2. Why rounds 32–34 all looked wrong, and it was not the fill

**The black crescent beside the highlight is the codec's ringing AROUND it.** AGENTS.md
said to leave it alone because it is coding undershoot (G and B exactly 0, R 8–26) and
"the artist reads it as the pupil". The first half of that is exactly why it had to go:
delete the bright lobe and keep the dark one, and what is left is a hard black arc round
a smooth fill. That is the shape the artist described as "weird brown stuff where the
softbox reflection used to be" in rounds 32, 33 and 34's first pass. Both lobes of one
artefact now go together (`soft_weight(..., rim=True)`), and the crescent is gone.

**AGENTS.md is wrong on this point and is corrected in this commit.**

## 3. Three released 2026 models, run on our own mask

All three take a mask. Ours replaces theirs: rounds 18–34 built a specular tracker for a
24-pixel iris, and SAM2 or a general highlight head is not going to beat it there.

| model | what it did | verdict |
|---|---|---|
| **SVOR 1.3B** (Xiaomi, CVPR 2026 winner) | default prompt: erased the iris, returned blank sclera. Prompted: a crisp photoreal eye with an **invented** pupil twice the size and a fresh catchlight. Two different prompts gave near-identical output. | **rejected** — object removal, and text is not real control |
| **EffectErase 1.3B** (Fudan, CVPR 2026 Highlight) | replaced the iris with a lump of shiny skin | **rejected** — same category error |
| **UnReflectAnything** (CVPR 2026) | removed the highlight and **kept her eye**: her brown, her limbal ring | **the right category**, used as material |

**The category is the finding, not the leaderboard.** SVOR and EffectErase are OBJECT
removal: the mask means "an object is here, delete it and rebuild the background", so a
mask on an iris deletes the iris. UnReflectAnything is specular removal: the mask means
"a highlight is here", and it returns the same surface without it. Do not reach for an
object eraser for this again.

**ProPainter** (2023, already in `docs/iris-prior-art.md` and never run until now) is
flow propagation, so it invents nothing — it moves her own observed pixels from frames
where the highlight is elsewhere. It works, with one non-obvious condition in §4.

**Licence, before any of this goes near a real video: ProPainter, SVOR, EffectErase and
UnReflectAnything are all NONCOMMERCIAL.**

## 4. ProPainter needs the occluder to MOVE between input frames

At stride 1 it put the catchlight **back** — peak 224 display against 215 in the source.
The highlight creeps about 0.34 px per frame in the eye's own coordinates at 59.94 fps, so
consecutive frames are nearly identical, nothing is ever uncovered, flow propagation has
nothing to propagate, and the model falls back on its prior — which, for a masked hole in
an eye, is a catchlight. At stride 5 the same model with the same masks takes the peak to
110–150.

Whether a given pass succeeds is **phase dependent**: at stride 5 the left eye's phases 2
and 3 came back at 0.95 and 0.96 of the source peak (highlight essentially intact) while
phases 0, 1 and 4 sat at 0.59–0.68. `eye_video_inpaint.py --strides 5,7,11` runs 23 passes
per eye and keeps the darkest result per pixel, because a failed pass leaves the region
BRIGHT.

## 5. The blocker: the tracked limbus is not on her iris

`scripts/audit_r34_track.py` walks outward from the tracked ellipse at nasal and temporal
angles (clear of the lids) and finds the iris→sclera step in the image itself.

| | left | right |
|---|---:|---:|
| observed limbus radius ÷ tracked radius, median | **0.936** | 0.997 |
| spread of that ratio | 0.127 (**1.66 px**) | 0.109 (1.42 px) |
| frame-to-frame change, p95 | 0.191 (**2.49 px**) | 0.168 (2.19 px) |

On a 12.9 px iris radius that is a **13 % systematic size error on the left eye** and
±1.5 px of wander with 2.5 px jumps between consecutive frames. Everything downstream is
anchored on this ellipse — rounds 32, 33 and every variant in this round — so the painted
iris inherits it, which is the artist's "double iris" and his "delayed, slow to react".

**Round 33 measured this landmark's jitter at 0.353 px rms and concluded it was solid.
That measured SMOOTHNESS against its own trajectory, not ACCURACY against the image.** A
smoothly wrong anchor is still wrong. Do not repeat that mistake: the control has to be
the photograph.

## 6. Measurements, and two of this round's own metrics that were wrong

**Temporal stability, per registered eye cell** (p5–p95 spread over frames, display
levels; a cell is the same piece of iris in every frame):

| | left | right |
|---|---:|---:|
| source control (her real eye moving) | 46.7 | 67.0 |
| ProPainter material + round 33 gain | 20.7 | 21.0 |
| UnReflectAnything material + round 33 gain | 21.1 | 24.0 |

**ProPainter and UnReflectAnything are equal on stability.** The artist's preference for
ProPainter therefore costs nothing and avoids the generation risk he named.

**The filled patch against its own surround**, frame to frame, with her own untouched
iris rotated 180° into the same ring as the control (round 32's frozen rotation design):

| | fill | her own iris | ratio |
|---|---:|---:|---:|
| left | median 2.71, p95 16.5 | median 4.68, p95 17.7 | 0.58× / 0.93× |
| right | median 2.75, p95 11.0 | median 3.31, p95 10.7 | 0.83× / 1.03× |

The patch varies no more than her photographed iris does. Chasing this further is chasing
the plate's own noise.

**Two metrics written this round were wrong, and both were wrong the same way — they
shared the bug's assumption:**

* A "fill/iris ratio traverse" reported 3–10× and looked damning. The footprint *sweeps*
  from rho .38 to .75, so it sits on genuinely darker iris early and brighter iris later;
  most of that traverse is real.
* A rolling-median "jitter" reported the fill as calmer than the source while the artist
  was plainly seeing colour change. It is a five-frame rolling median, so it discounts
  slow drift by construction — and slow drift was what he was watching.

Anything temporal must be measured **in the registered eye frame, against a paired
control taken from her own pixels.**

## 7. The illumination gain, replaced

Round 33's per-frame field is a constant plus a linear ramp per channel, fitted in the log
domain and clamped at exp(±2.5) — a **12-fold** multiplicative range — fitted on clean
iris and then evaluated INSIDE the footprint where it has no evidence. `local_gain()`
replaces it with one robust ratio per channel from the observed iris ring around the
footprint, bounded 0.5–2.0. It tracks the moving lid shadow for the same reason the ramp
was introduced (the evidence is right there) and cannot extrapolate.

## 8. The scene's own catchlight — built and validated, then PARKED

`render_eye_catchlight.py` puts a corneal mirror in the shot's actual environment and
lets Cycles find it. Validated against an independent numpy model of the same reflection
sharing no code (`validate_eye_catchlight.py`):

| variant | NCC | peak offset px | 
|---|---:|---:|
| **Blender convention, prefiltered** | **0.942** | **0.0 median, 4.47 max** |
| panorama flipped in u | 0.077 | 14.66 |
| panorama flipped in v | 0.007 | 12.10 |
| normal instead of reflection | −0.012 | 7.25 |

Three things it corrects in stage 11d, all measured:

* **The cornea is a CAP, not a sphere.** A full mirror sphere has a grazing rim at 100 %
  Fresnel sitting exactly where the sclera and lid are; rendered whole, the left eye's
  brightest pixel was 1.576 linear on that rim. Truncating at the limbus
  (sin θ = 5.85/7.8) removes it by construction.
* **The proxy must NOT be the occluder.** With it in, frames 20–22 keep only 22–36 % of
  the corneal reflection energy: the proxy is a closed silhouette shell with no eye
  sockets, so it buries the cornea rather than shading it.
* **The cornea is anchored on the tracked iris, not the proxy apex** — which moved it a
  median of 0.90 mm, p95 1.47, max 1.72 (`corneal_anchor.py`, shared by renderer and
  validator so they cannot place different spheres).

Compositing is through the eyelid polygon AND the limbus. `verify_r34.py` confirms
**110,213 reflection pixels inside the eyelids and 0 outside** — the defect that got
stage 11d rejected.

**The artist parked it:** *"don't start adding our own catchlight reflection stuff before
we have fixed this properly."* Respect that. But record what the face-scale frames show:
with the softbox gone and nothing in its place, the eyes read as dead at the scale the
shot actually plays. Removal alone has a ceiling below the bar.

## 9. A regression I caused, recorded so it is not repeated

Two "improvements" made the result worse and the artist caught both:

* `--atlas-mode holes` (use her observed pixels, model only in never-seen cells) put the
  round-33 source-pooled material *back* for most cells — the brown mush he had already
  rejected. Those cells are exactly the ones with the fewest and poorest observations,
  which is why the model's reconstruction beats them.
* `--union-footprint` raised weighted pixels from ~40 to 65/76 per eye, spreading that
  material further.

Both remain available and both default OFF for the whole-iris path.

## 10. Verification

`logs/r34_verification.json` for `r34_newiris2`: 150 frames, **310,951,026 pixels
bit-exact source outside the reported weight**, every written source frame identical to
the keyed plate, all 274 treated eyes darker than source, and the standing **300 exact
keyed-eye fallbacks and 304 protected hashes** pass. `production_enabled` and
`artist_accepted` are false.

```powershell
# the eye handed to a model, using our mask
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_video_inpaint.py --shot music_videos/reporter_test/shots/reporter_a --export --strides 5,7,11
#   ... run ProPainter over each dependencies/... phase dir ...
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_video_inpaint.py --shot music_videos/reporter_test/shots/reporter_a --collect --strides 5,7,11
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_video_inpaint.py --shot music_videos/reporter_test/shots/reporter_a --apply --strides 5,7,11

# UnReflectAnything's DIFFUSE decoder, which round 19 never ran
& dependencies/unreflect-runtime/.venv/Scripts/python.exe -B scripts/unreflect_diffuse.py --frames <crops> --masks <masks> --out <dir>

# ONE canonical iris, rendered into all 150 frames
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_soft_removal.py --shot music_videos/reporter_test/shots/reporter_a --name r34_newiris2 --atlas-frames <dir> --atlas-mode replace --fill-iris 0.95 --grain 0 --pupil-target 0.34
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/verify_r34.py --shot music_videos/reporter_test/shots/reporter_a --name r34_newiris2
```

Reviews (display-sRGB CRF 12, review only, not masters and not acceptance):
`final/r34_newiris_slow.mp4` is the approach the artist endorsed;
`final/r34_best_slow.mp4`, `final/r34_unreflect_panel_slow.mp4` and
`final/r34_inpaint_panel_slow.mp4` are the rejected intermediates, kept because the
comparison is the argument.

## 11. Starter prompt for round 35

> Stay on eyes. Read round34.md, then round33.md. **The construction is settled and the
> artist endorsed it: the eye is generated ONCE, in the eye's own registered frame, and
> rendered into every frame. Do not go back to anything refit per frame — that is what
> killed rounds 31, 32, 33 and three of round 34's four attempts.** Do not re-evaluate
> SVOR or EffectErase: they are object-removal models and they delete the iris; §3 has
> the evidence. UnReflectAnything and ProPainter are equal on stability (§6) and either
> can supply the canonical material.
>
> **The blocker is the tracker, and it is measured.** The mediapipe four-point limbus
> ellipse sits at 0.936 of the observed iris radius on the left eye with 1.66 px of
> spread and 2.49 px p95 frame-to-frame change against the image (§5). That is the
> artist's "double iris" and his "delayed, slow to react". **Fit the limbus to the IMAGE
> EDGE each frame** — `scripts/audit_r34_track.py` is most of the detector already — then
> smooth that trajectory, and feed it to everything: the atlas registration, the painted
> disc, the lid matte and the corneal anchor. Round 33's 0.353 px figure measured
> smoothness against its own trajectory, not accuracy against the photograph; do not
> quote it as accuracy again.
>
> Then: the pupil is still not black enough (`--pupil-target` darkens the canonical iris
> centre to the darkest level her own plate shows, measured, but it needs checking on
> frames), and the canonical iris has no fibre texture — it only has to look right ONCE,
> so that is now a tractable problem rather than a per-frame one.
>
> The scene catchlight is built and validated (§8) and the artist PARKED it until removal
> is clean. Do not switch it on unprompted. But the face-scale frames say removal alone
> reads as a dead eye, so raise it with him once the tracker is fixed.
>
> **Working method, and this round violated it repeatedly:** put frames in front of him
> constantly, inspect every frame at eye scale, and **look at frames before and after
> every change** — two of this round's last three "fixes" made the result worse and he
> caught both before any metric did. Report p95 and max, never a median alone, and
> measure temporal behaviour in the registered eye frame against a paired control taken
> from her own pixels; §6 records two metrics written this round that were wrong because
> they shared the bug's assumption. Preserve exact keyed fallback, the 300/304 protected
> checks, generic input support and observed identity. All four models are
> NONCOMMERCIAL. No key retuning, no capture requirements, no impossibility claims.
