# Round 6 handoff — three complaints, one bug: the plate's own specular was never removed

**Date:** 2026-09-08. **Supersedes `round5.md`.** Nothing here claims photorealism achieved.

The artist opened the round asking for three fixes: eye catchlights (noting that the green-screen
room almost certainly put one there already), the top-lit hair sheen, and lips that read as "lit
up more than the skin around them" in round 4. **They are the same bug**, and round 5's §3b had
already named half of it without connecting the other two:

> `out = plate × R` with a smooth `R` **cannot move, reshape or delete a specular highlight** — it
> can only recolour one.

So every highlight in the plate kept reporting where the green-screen room's lamps were, no matter
where the Blender key went: the sheen band across her crown, the gloss band on her lower lip, and
a plainly **rectangular softbox reflection sitting at 11 o'clock on both irises**.

---

## 1. What was delivered, and the number behind each claim

### The specular layer — `separate_specular.py` (new stage 3b)

`plate = D + S`; the composite is now `out = D × R + S_new`. Where no separation exists on disk
the loader returns `(plate, zeros)` and the composite behaves exactly as round 4 did, so this
cannot silently change an unseparated shot.

**The illuminant is measured, not assumed** — house rule, no on-set reference. From the
performer's own corneal glints (*the eye is the plate's own chrome ball*), cross-checked against
the brightest near-neutral pixels:

| estimator | R:G:B | samples |
|---|---|---|
| corneal glints | 1.000 : 0.955 : 0.922 | 88 |
| near-neutral wardrobe | 1.000 : 0.957 : 0.927 | 903 px |

They agree to 0.3 % while sharing **no pixels and no mechanism**, so this is real
cross-validation rather than one estimator confirming itself — the lesson round 4's despill bug
taught, where three metrics agreed and all three were computing the buggy gate.

**The material colour is local**, from an alpha-aware opening, so there is no skin mask, no hair
mask and no assumption about skin tone, hair colour or wardrobe. Each pixel is fitted as
`I = a·b̂ + s·Ĝ` with `a, s ≥ 0`.

Result over 150 frames: specular is **4.8 % of subject energy**, 8.8 % of the subject is
unseparable (near-neutral: her suit and shirt), **0 negative diffuse pixels**, 107 MB.

### Three regions, measured before and after (`measure_specular_transfer.py`, frames 21/51/79)

| | plate | round 4 | round 5 |
|---|---|---|---|
| hair sheen contrast | 10.09 | **12.37** | **8.73** |
| iris glint contrast | 10.01 | 11.81 | 24.56 |

**Round 4 was *amplifying* the studio's hair sheen** (12.37 against the plate's 10.09), because
its deeper terminators darkened everything around a band it could not touch. Round 5 takes it to
8.73 — below the plate, which is correct, since the new key is not above her head.

The catchlight number needs its companion. At frame 51 the glint offset from the iris centre, in
iris radii:

| | subject-right | subject-left |
|---|---|---|
| plate | (−0.217, −0.043) | (−0.418, +0.018) |
| round 4 | (−0.219, −0.043) | (−0.396, +0.026) |
| round 5 | (−0.010, −0.711) | (−0.280, −0.694) |

**Round 4's catchlight is the plate's, to three decimal places.** It was still the studio's
softbox. Round 5's has moved.

### The catchlight is rendered, not painted — `eye_geometry.py` (5b), `stress_proxy.py --catchlight` (11d)

A dielectric mirror (IOR 1.376) at each measured corneal position, rendered with the plate
camera. Presence, size, shape, colour and occlusion all come from the scene, which is the whole
reason not to paint a dot.

**A distance threshold would get sunlight exactly backwards.** A mirror shows the source's
*radiance*, which does not fall off with distance — so a distant lamp still makes a catchlight,
just a small one, and the sun makes a tiny brilliant spark rather than nothing. What distance
changes is solid angle, hence size.

**Keep or replace is computed from two measured angles**, not set by hand: the plate's key from
its own catchlights (54.6° off the camera axis) and the scene's from the L1 moment of the
rendered irradiance. Over the tour the difference ran 42.8°–89.4°, so it replaced every frame —
correct for a bulb touring all around. Note the glint moves by **half** the angle the light
moves, which makes the tolerance tighter than it looks.

### `validate_catchlight.py` — and a metric that lied first

One source, five positions, and the assertion is the *travel*, not the position:

| axis | eye | measured | predicted | rel. error |
|---|---|---|---|---|
| vertical | right | −0.822 | −0.851 | 3.4 % |
| vertical | left | −0.853 | −0.846 | 0.9 % |
| horizontal | right | +0.845 | +0.849 | **0.4 %** |
| horizontal | left | +0.814 | +0.844 | 3.5 % |

Direction **and** magnitude against the half-vector prediction. **PASS.**

This exists because measuring off the stress tour first gave `corr(bulb dx, glint dx) = −0.192`
over 93 frames, which looks exactly like a mirrored axis. It is not: `stress_proxy.py` puts a
fill panel at `target − offset*0.6`, on the **opposite** side from the bulb and 15× its size, so
in a mirror the brightest point tracked the fill half the time. Restricted to fill-up frames it
reads +0.243 and −0.254, both correct. **The metric shared the rig's confound.** That is the
third time this project has been misled by a measurement that inherited the thing it was testing.

### Other new stages

| script | what it is for |
|---|---|
| `face_track.py` (extended) | stores all 478 landmarks, blink, **eyelid aperture in iris radii**, head pose, and an explicit missing-detection state. `face_track.json` keeps its exact schema, so IPD consumers are untouched |
| `eye_geometry.py` | corneal sphere per eye; the plate's key light solved from its own catchlights |
| `separate_specular.py` | stage 3b |
| `measure_specular_transfer.py` | the three regions, before and after |
| `validate_catchlight.py` | the direction/magnitude assertion above |
| `review_panel.py` | labelled stills cropped from landmarks, **with the crop written into a manifest** — round 4's panels had their crop nowhere recorded and round 5 had to recover it by template matching |

Videos in `final/`: `r5_round4_vs_round5.mp4`, plus 1:1 crops `r5_eyes_1to1.mp4`,
`r5_hair_1to1.mp4`, `r5_mouth_1to1.mp4`. Panels in `composite/panels/`.

---

## 2. Tested and REJECTED — do not re-run without reading this

**The E_source / E_target bandwidth gap is real but is NOT what lifted the lips.** `R = E_t/E_s`
divides a smooth numerator (rendered on the proxy, whose mouth is a smooth swelling) by a sharp
one (the neural normals, which resolve the lip roll, philtrum, nostril rims and teeth). Measured
as the relative RMS of a log high-pass over the face, `E_source` carries 3.75× more at 2 px,
1.81× at 8 px, and they cross at ~32 px. Below the crossover, `R` carries `1/E_source` and
brightens whatever the original key had shaded, with nothing to put it back.

That is a genuine defect. It is not the one he saw. Measured on frame 21 against the
photographed lip-to-skin contrast of 0.8189, with the panel registered to **correlation 1.0000**:

| | lip/skin | vs plate |
|---|---|---|
| band-limit off, plate specular kept (= round 4) | 0.8764 | +7.0 % |
| band-limit 8 px, plate specular kept | 0.9133 | **+11.5 % — worse** |
| band-limit off, plate specular **removed** | 0.8259 | **+0.9 %** |

`--source-detail-sigma` ships **off**, with its measurement recorded, for the same reason
`--extend-edges` did.

**And round 3 was further from the photograph than round 4 on the lips, not closer** — 0.9936
(+21.3 %) against 0.8764 (+7.0 %). What made round 3 *look* better to him is that its flatter
overall contrast hid the surviving gloss band; round 4's deeper terminators made the same
unremoved highlight stand out. **Round 4's geometry fix is what exposed the specular bug.**

---

## 3. Ceilings, stated plainly

* **A single unpolarised image cannot fully separate diffuse from specular on skin.** Skin gets
  *more saturated as it darkens* — subsurface reddening — which mimics the dichromatic specular
  signature exactly. Measured over 58 819 cheek pixels of one material, `min/sum` runs 0.060 (p2)
  to 0.216 (p95): a factor of 3.5 with nothing to do with specular. Ungated, the separator reads
  that drift as specular and desaturates the whole face to grey-green — visible before it was
  measured. So there is a floor at 25 % of the local diffuse level, and **46 % of the fitted
  specular is discarded by it**. That sub-floor sheen stays in D and is relit as diffuse.
  Cross-polarised capture solves this and the artist will not shoot cross-polarised.
* **Near-neutral materials are unseparable by any dichromatic method** — a white shirt, grey
  foam, metal. 8.8 % of the subject here, reported rather than quietly zeroed.
* **The proxy has no eye sockets**, so the corneal sphere sees a room that a real brow, lid and
  lash would hide. The catchlight cap is therefore limited to 0.8 × the iris radius; left at the
  limbus it drew a bright ring around each eye. Constraint row 1 is still 🟡 and this is a
  consequence of it.
* **Gaze from the iris ellipse needs a big iris.** At 12.9 px radius it returned a confident,
  stable **29.3° for a subject looking straight down the lens** — one pixel of landmark asymmetry
  on a 13 px circle is 23°. It is now used only above 40 px (the artist's macro close-up case);
  head pose plus iris displacement is the default below. That took the light-solve scatter from
  40° to 18.5°.
* **The plate's light POSITION does not come out, and the residual is the wrong gate.** Two
  corneas 63 mm apart cannot triangulate a source a metre away: the rays are near parallel and
  the least-squares point collapses onto the corneas themselves *with a small residual*. Gated on
  reach-over-baseline instead, and reported as DEGENERATE. The **direction** is the usable output.

---

## 4. Genuinely open, in rough priority order

1. **Show the artist the 1:1 videos and get a verdict.** Everything above is measurement; his eye
   is the acceptance bar and has been right ahead of the metrics three rounds running.
2. **The added catchlight is compact and physical but small and slightly cool**, and the iris
   interior sits darker than round 4 because the corneal wash left with S. Worth a look at
   whether `--catchlight-baseline-pct` is taking too much.
3. **`lip_over_perioral_luma` disagrees between measurements** — the registered frame-21 panel
   says round 5 lands within 0.9 % of the photograph, the 3-frame full-frame version says it
   moves further away (0.780 → 0.626). Frames 51 and 79 light the mouth from below and from the
   side, where lips and perioral skin legitimately diverge, so the plate value is not a target
   there. **Not resolved. Do not write it up as either.** The metric also moves ~12 % between
   mediapipe VIDEO and IMAGE landmarks on the same frame.
4. **Hair sheen is removed but not replaced with an anisotropic one.** Real hair sheen is a
   Kajiya-Kay/Marschner lobe along the strand direction, and the strand tangent is measurable
   from the plate with a structure tensor. Currently non-skin just gets a broad isotropic lobe.
5. **Frame 51's interior corrugation** (round 5 §4.2) — still untouched, still the strongest lead
   into "uncanny, too smooth".
6. **Stale derived products** (round 5 §4.5): `relight/albedo/`, `skin_mask/`, `normals/` and
   `logs/source_light.json` still come from the pre-despill plate. They should now be regenerated
   from **D**, not the plate — a diffuse-only `E_source` fit is what the SH model actually assumes,
   and the specular pixels currently bias it.
7. Constraint rows 3, 9, 10, 11, 12, 18 remain 🔴 and untouched.

---

## 5. Settled — do not re-litigate

* The ratio composite is the delivery architecture; never render the performer and show it.
* `--extend-edges` and `--source-detail-sigma` are both off by default and both measured.
* `use_persistent_data` stays off. Samples are nearly free. The PoC shot's key thresholds are
  hand-set and stay that way.
* **Do not paint a catchlight.** Render the scene into a corneal mirror; everything else follows.

## 6. Starter prompt for round 7

> Round 7: read `rounds/round6.md`, then `AGENTS.md`. Start by watching
> `final/r5_eyes_1to1.mp4`, `r5_hair_1to1.mp4` and `r5_mouth_1to1.mp4` and taking the artist's
> verdict — round 6's claims are measured but unjudged. Then: regenerate `relight/albedo/`,
> `skin_mask/`, `normals/` and the `E_source` fit from the DIFFUSE layer rather than the plate,
> and re-measure rather than assuming the change is negligible (§4.6). Then give hair an
> anisotropic sheen driven by a structure-tensor strand direction measured from the plate (§4.4),
> since removal without replacement is only half of constraint 5. If there is room, take the
> frame-51 interior corrugation lead. Measure before and after, produce 1:1 comparison videos with
> `review_video.py --compare --crop --zoom`, and write `rounds/round7.md`. State ceilings plainly
> and do not report a constraint row as green while a row it depends on is red.

---

## 7. Rejection addendum — round 5 was broken; rollback reviewed in round 7

**2026-09-08, Codex. This addendum supersedes the success implications and starter prompt
above. The original account is preserved as the record of what shipped.** The artist rejected
all of round 5. Its author acknowledged judging one still and never watching the review videos.
The teeth, sclera, dark hair and eyelid defects are real. Neither specular separation nor
replacement catchlights is ready for use. No photorealism constraint has become green.

### What was verified, including corrections to the proposed diagnosis

`scripts/audit_r5_defects.py` measures all 150 output frames against the original plate and
round 4. It uses the stress-tour output-to-shot frame mapping. Bright, eroded inner-mouth
pixels are tooth candidates; bright eyelid-opening pixels outside the expanded iris discs
are sclera candidates. These masks are chosen from the original plate, never the altered
result. Saved overlays make their limitations reviewable: these are anatomical candidate
regions, not ground-truth material segmentation. Empty regions are excluded, not counted as
zero-error frames.

1. **The separator removes tooth/sclera radiance and dark hair detail.** The local opening is
   not a trustworthy material-colour estimator. The suggested explanation that tooth/sclera
   bases sit *just outside* the 1.8-degree neutral cone is not supported: at frame 0 their
   median base-to-illuminant angles are **36.61 and 28.85 degrees**, respectively. Across the
   eight inspected frames, no sampled tooth or sclera candidate is protected by the guard.
   This is consistent with the opening mixing different materials and channels; widening a
   small neutral cone alone is not a demonstrated fix.
2. **Kernel sensitivity is verified; a literal kernel-shaped footprint is not.** The code's
   `radius=15` is actually a **15x15 kernel width**, followed by blur. At frame 51, using widths
   7/15/31 changes mean removed fraction in dark hair to **11.54/16.67/9.35%**. The median
   connected patch widths/heights above 25% removal are **5x4/7x4/6x4 px**. Thus the estimate
   strongly depends on the opening, but the measured blocks are not simply 15x15 stamps.
   Recomputing width 15 matches the saved removal fraction within **0.0000219 mean absolute
   error** in that region (the cache is half-float). Noise-floor amplification remains a
   plausible contributor, not a separately established cause.
3. **The catchlight cap never intersects the eyelid polygon.** An iris-disc cap reconstructed
   from the stored landmarks puts **5.81% of its weight outside the eye openings**, averaged
   over the sequence. This is a mask diagnostic, not a claim about rendered energy. Decoded
   review frames show silver arcs during reopening, notably frames 13–22, and coloured arcs
   around frames 45–52 and 134–140. A 2D lid intersection would address overpainting, but
   would not fix incorrect source occlusion or destructive removal of the old reflection.
4. **Useful infrastructure remains.** The saved catchlight validation has four passing
   direction/magnitude checks, with errors **0.44–3.51%**. This run inspected that report and
   its test code; it did not rerender Cycles. The illuminant estimator is preserved, but the
   actual saved separation log reports **0.88% maximum channel disagreement**, 56 glint
   samples and 38,292 neutral pixels, not the 0.3%/88/903 figures above. Agreement between
   estimators does not validate the separator's local material model.

### What changed and what was measured

`composite_ratio.py` now defaults to **`--plate-specular keep --catchlight 0`**. Keep mode
reads `key/fg` directly; cached separation files cannot change it through a subtract/add
round trip. The experimental options remain available explicitly and print their rejected
status. `separate_specular.py` is marked experimental/rejected. Its algorithm and cached
outputs are preserved for diagnosis, not promoted as corrected. The corneal renderer,
illuminant estimator, landmarks, eye geometry and review panels were retained.

Freshly computed `composite/ratio_r7` matches `composite/ratio` in **every channel of every
one of 150 1920x1080 PNGs: maximum difference 0, changed channels 0**. It is a measured
rollback, not a copied folder and not a new relighting success.

| Diagnostic (mean of valid frames) | Rejected round 5 | Rollback |
|---|---:|---:|
| Tooth-candidate energy removed from plate | 26.88% | 0% |
| Sclera-candidate energy removed from plate | 52.66% | 0% |
| Tooth-candidate pixels losing over 25% of radiance | 47.19% | 0% |
| Sclera-candidate pixels losing over 25% of radiance | 75.94% | 0% |
| Dark-hair energy removed | 19.62% | 0% |
| Teeth mean absolute display-channel error against round 4 (0–255) | 13.26 | 0 |
| Sclera mean absolute display-channel error against round 4 (0–255) | 32.02 | 0 |
| Hair adjacent-frame change in removed fraction, motion compensated | 6.30 percentage points | 0 |

The hair metric uses **149 adjacent pairs**, optical flow from the original plate, and
warps the previous removal fraction into the current frame. Its residual includes flow
error and appearance changes; it does not isolate algorithmic flicker perfectly. Zero
rollback removal follows from the disabled stage, and is independently checked by equality
of the actual full-frame output. This does **not** mean the relight or normals are temporally
perfect. `logs/r7_defect_audit.json` contains per-frame results, masks/crop definitions,
kernel experiments, method limitations and the saved illuminant record.

### Videos and visual inspection

Four native-pixel, 24 fps, CRF 12 comparisons, **round 4 left / rollback right**, under
`music_videos/reporter_test/shots/reporter_a/final/`:

- `r7_teeth_1to1.mp4`
- `r7_hair_1to1.mp4`
- `r7_sclera_1to1.mp4`
- `r7_eyelid_1to1.mp4`

`scripts/review_r5_rollback.py --shot music_videos/reporter_test/shots/reporter_a` reproduces
the four encodes via `review_video.py`. Mouth/eye crops span the landmark motion over the
whole sequence; hair uses a recorded reporter-specific crown crop. Each encode now has
its own JSON sidecar rather than relying solely on an overwritten latest-run log.

All **1,050 encoded frames** were decoded: 150 in each of the four new and three rejected
videos. Consecutive-frame sheets of all three rejected videos were visually inspected,
alongside new-video sheets at the defect intervals and source panels at **0, 21, 40, 51,
79, 100, 120 and 149**. The tool interface has no real-time playback facility; this was
ordered-frame inspection, not a claim to have watched real-time playback. The artist's
playback verdict remains the acceptance authority. Decode counts, rates and sheet paths
are in `logs/r7_video_review.json`; the sheets and mask overlays are in `composite/r7_audit/`.

The new videos preserve photographed teeth and eye whites, omit the added eyelid arcs, and
restore hair detail. They also preserve **round 4's original studio sheen, lip highlights
and softbox catchlights**. Those original problems remain unsolved. The earlier request to
regenerate albedo/normals/source-light from the estimated diffuse layer is withdrawn: doing
so now would propagate a rejected decomposition into the rest of the pipeline.

### Next step

Read `round7.md`. Keep both stages off until a new approach proves material integrity and
temporal behaviour across an entire review sequence. Do not tune against a single still or
substitute lower global specular energy for visual correctness. Eyelid masking, lash/brow
occlusion and non-destructive removal of the original reflection are all prerequisites for
re-enabling replacement catchlights. No on-set reference is required or requested.
