# Round 32 handoff — same-frame anatomical completion; rejected for colour flicker

**Date:** 2026-09-09. Starts from round 31. Production defaults unchanged.
**Delivered:** a same-frame anatomical completion of the corneal highlight over all
150 frames, contact sheets covering every frame, a withheld-pixel evaluation against
three paired controls, a measured flicker diagnosis, two source/repair videos, and
exact-source verification.
**Not delivered:** an acceptable eye repair. **The artist rejected the result on
sight.** No scene-light replacement, no temporal appearance model, no production
change. Photorealism is not met.

## 1. Artist steering and rejection

The artist rejected the round in these terms:

> It looks terrible and flickery on the repair eyes slow video. The color within
> the pupil keeps changing. This is not acceptable.

He had earlier rejected an intermediate state with a specific and correct
observation, which redirected the round:

> There are plenty of problems in that one with pupils that are almost uniformly
> brown with no black within them. Compare that with the source references next to
> them, where there's a clear black pupil behind the softbox

He also set two working requirements that are worth carrying forward
independently of this round's outcome:

* **Show frames constantly, not at the end.** Judgement happens on images.
* **Inspect all the data.** He pointed at `composite/r29_shape/contact_120.jpg` as
  the standard: every frame on a sheet at eye scale, not four chosen frames. That
  standard immediately exposed a failure mode that a four-frame sample had hidden,
  and it should be the default for any eye round from here.

Selective generation remains authorized from round 30; this round did not use it.
Nothing here reinstates any earlier ban.

## 2. What was built

`scripts/eye_reflection_repair.py` — removes the capture room's corneal highlight
and completes the footprint from the **same frame's own eye**. No generated
material, no cross-frame correspondence, no fitted-and-subtracted layer.

The method, per eye per frame, inside a local crop:

1. Limbus ellipse from the four mediapipe cardinal iris landmarks, checked against
   the plate at 24× before being trusted.
2. Appearance model = per-channel **median radial profile** (bins of 0.055 out to
   ρ = 0.88) times a **smooth angular shading gain** (constant plus two harmonics,
   fitted on per-angle medians).
3. Footprint = signed-residual hysteresis against that model, seeded at the
   brightest unexplained pixel, confined within 7 native pixels of the seed.
4. Completion = the model plus an angular interpolation of the **residual**, taken
   from the same frame at the same normalized radius, with a two-pixel donor collar.
5. Abstention keeps bit-exact source.

`scripts/audit_r32_repair.py` scores it on withheld pixels. `docs/r32-eye-protocol.md`
was frozen before any error number existed; **§4 below records the amendments made
after that freeze**, which is why its own text no longer matches the shipped method.

## 3. Measured findings

### 3.1 Coverage

280 of 300 eye instances repaired across 150 frames; 17 abstain as closed or
occluded (the blink at source 707–711 and the second blink), 3 for an insufficient
donor annulus. Footprint size: median 53, p95 77, max 87 native pixels.
**14,422 pixels changed in total across the whole 150-frame sequence.**

### 3.2 Withheld pixels — the fill is better than round 31's, and not good enough

The real footprint of each repaired eye is rotated about the limbus in the eye's own
normalized polar frame, which preserves its size, shape and radius exactly; the
rotated copy is then withheld and predicted. 1,400 sites attempted, **273 scored**
(1,091 rejected for overlapping the real footprint or its collar — at ρ ≈ 0.3–0.6 a
footprint's angular width leaves little room for a disjoint copy), 12,082 pixels.

Median absolute display RGB error, 0–255 levels, identical pixels in every column:

| fill | left | right |
|---|---:|---:|
| **round 32 repair** | **9.4311** | **8.6855** |
| profile only, no angular residual | 9.7077 | 8.9125 |
| annulus median | 14.0009 | 14.1653 |
| flat ring gain — round 31's fill shape | 12.9979 | 15.0028 |
| plate's own angular texture variation (floor) | 4.9442 | 5.4481 |

The repair beats round 31's fill shape by 27.4 % left and 42.1 % right on identical
pixels, and beats a naive annulus median by 32.6 / 38.7 %. It sits at **1.91× / 1.59×
the plate's own texture floor**, so it is not at the level of an observed pixel.

**The angular residual is worth almost nothing**: 9.4311 against 9.7077 for the
radial profile alone. The completion is essentially the profile. Record this as a
negative result — the same-radius donor idea did not pay for itself.

### 3.3 Why it was rejected — measured

`logs/r32_flicker.json`. Mean absolute change of the filled region's display colour
between adjacent repaired frames, and the same pixels in the source as a control:

| | left | right |
|---|---:|---:|
| adjacent frame pairs | 140 | 134 |
| repair step, median | 2.882 | 2.123 |
| repair step, p95 | 16.243 | 12.175 |
| repair step, max | 58.765 | 52.151 |
| source step, median (control) | 4.649 | 4.175 |
| source step, p95 (control) | 17.079 | 17.203 |

Fill display colour across the shot: left R 36.23→131.93, G 12.04→87.30,
B 2.20→65.52, median [60.45, 37.35, 22.81]; right R 34.35→122.90, G 18.61→81.42,
B 6.14→66.43, median [62.28, 36.59, 20.18].

Relative to its own surround (`logs/r32_withheld.json`), the fill/surround ratio has
median [0.6227, 0.5523, 0.4210] left with IQR [0.2378, 0.2638, 0.3018] and max/min
[6.07, 11.54, 42.16]; right median [0.6077, 0.5302, 0.5020], IQR
[0.1610, 0.1995, 0.2347], max/min [3.62, 5.88, 10.75]. R/B ratio median 1.5885 left
and 1.2261 right, max/min 12.12 and 4.47.

**The median step is smaller than the source's own, and that is exactly the trap.**
A median below the control looks like stability and is not what the eye judges: the
p95 and max are where the artefact lives, and the fill's colour traverses a 3.6×
range in red and a 30× range in blue over the sequence. A 50-pixel patch inside a
24-pixel iris jumping by up to 59 display levels between adjacent frames reads as
flicker no matter what the median says. **Do not use a median adjacent-frame step as
a temporal-stability gate again; report p95, max and the full range.**

### 3.4 The structural cause, which is the useful part

**Every term of the fill is refit from scratch on each frame.** The radial profile
is a median over a few dozen pixels of that frame; the angular gain is fitted on that
frame; the residual donors are that frame's pixels. Nothing carries across frames.
So although this round fits no explicit per-frame gain — which is what round 31 did
wrong — it reproduces round 31's symptom through a different mechanism.

**A per-frame-refit fill cannot be temporally stable at this scale, however it is
derived.** That is the round's main transferable finding. Any next attempt needs the
eye's appearance estimated **once over the sequence** in a stabilized coordinate
frame, with only a smooth, explicitly-constrained per-frame illumination term — not a
model rebuilt every frame from a few dozen pixels.

### 3.5 Three implementation bugs, all initially mistaken for threshold problems

Recorded because each looked like a tuning issue and none was:

1. **The model region reached past the limbus** (ρ < 1.18). The sclera is bright and
   sits left and right of the iris but not above and below it, so it is not radially
   symmetric about the iris centre. Profile and angular gain were being fitted to
   iris, sclera and lid together; residuals were large everywhere and **106–197 of
   about 460 visible pixels** tripped the 4σ level while the highlight itself did not
   stand out. No threshold could have fixed that. Model now stops at ρ < 0.88, just
   inside the limbal ring, whose radius varies with angle and smears across bins.
2. **The angular shading gain was ordinary least squares on the ratio.** The
   highlight is a dozen pixels at ratio ≈ 10 to the profile, so it dragged the whole
   shading model upward, the model over-predicted the rest of the iris, and the
   resulting **negative** residual across the upper iris — not the highlight — was
   what the detector flagged. Fitting per-angle medians instead dropped the flagged
   count from 87–131 to 48–59 per eye.
3. **The footprint guard was a fraction of visible area** (25 %). Twenty-seven eyes
   abstained with footprints of 27–101 pixels — the same absolute size as accepted
   ones — purely because the eye was half shut and the denominator small. That
   produced removal on one frame and not the next. The guard is now absolute.

Also: unlimited connected growth through a low hysteresis level floods along iris
texture (67–118 px on most frames); growth must be bounded and must use the residual's
**sign**, because the highlight is a positive core, the coding undershoot beside it a
negative rim, and iris texture is both signs at random.

### 3.6 Two claims this round made and then had to correct

Both are recorded so they are not inherited as facts.

* **"The pupil edge is not resolvable in this plate."** Reported after measuring a
  smooth radial profile with no step. **That was an artefact of profiling about the
  limbus centre**, and the artist caught it by looking at the source. The corrected
  statement is narrower and still negative: once the highlight **and its coding
  undershoot** are excluded, a pupil does not localize either — across six eyes the
  fitted centre scatters over the whole ±4 px search range on 6–10 pixels of
  evidence, with no step in the profile.
* **Centring the polar frame on the fitted dark core.** Tried, and wrong: that core
  is the highlight's own undershoot rim, so centring on it put the highlight at small
  radius where it dominated its own profile bins and hid itself. The mask came out
  shifted off the highlight and left a bright speck inside the dark region. Reverted
  to the limbus, which is the only stable anchor this plate carries.

### 3.7 What the black around the highlight actually is

The plate is tagged `bt709` limited range and was expanded on ingest
(`logs/extract_command.json`, `in_range=limited out_range=full`). The undershoot
immediately beside the highlight reaches **exactly 0 in green and blue** while red
stays at 8–26/255 — the signature of a linear filter's undershoot clipped at legal
black, not of a brown pupil. It is nonetheless the only dark material the plate
carries around the pupil, and the artist reads it as the pupil. The shipped method
therefore **does not repair it**: it removes only the positive core and leaves that
black observed. Replacing it is what produced the uniformly brown pupils he rejected
in the intermediate state.

Related measurement, contradicting an assumption worth not re-deriving: **the
highlight is not clipped.** Peak display luminance inside the limbus is 0.7843–0.8823
across source 700–723 with **zero pixels at 255**. The corrupted region is a sum, not
a sensor hole.

## 4. Amendments made after the protocol freeze

`docs/r32-eye-protocol.md` was written before any error number existed and is
retained unchanged. The shipped method then diverged from it in three ways, all
after development-frame inspection and **before** any withheld-pixel score was
computed:

1. §3.1–3.3 describe a model out to ρ = 1.18; shipped is ρ = 0.88 (§3.5.1).
2. §3.4 describes symmetric hysteresis on the absolute residual; shipped uses
   sign-aware bounded growth, and **does not repair the dark rim at all** (§3.7).
3. §2.3 and §3 state the frame is anchored on the limbus because no pupil is
   resolvable; that reasoning is corrected in §3.6 above. The anchor is unchanged.

The withheld-pixel design in §4 of the protocol ran exactly as frozen. Its rotation
scheme rejects 1,091 of 1,400 sites for self-overlap, which was not anticipated and
is a design weakness rather than a result.

## 5. Verification

- **300 exact keyed-eye fallbacks and 304 protected hashes pass** (reusing
  `verify_r15.fallback`).
- All 150 written frames: source bit-exact outside the mask, all values finite and
  non-negative, 0 failures.
- No production module changed. `production_enabled` and `artist_accepted` are false
  in `logs/r32_repair.json`.

Reviews, display-sRGB CRF 12, not masters and not acceptance:

- `final/r32_repair_eyes.mp4` — 150 frames at 60000/1001 fps, source left / repair
  right, eye band cropped to `848 204 265 178` at 4× nearest-neighbour.
- `final/r32_repair_eyes_slow.mp4` — the same at 12 fps. **This is the file the
  artist rejected.**
- `composite/r32_repair/sheets/contact_{000,025,050,075,100,125}.jpg` — every frame,
  both eyes, source and repair side by side with the mask outlined.

`composite/r32_repair/` retains `source/`, `repair/`, `mask/`, `sheets/` and
`repair.json`. The full-frame `linear/` EXRs and the wide `review/` tiles were
deleted to keep the round at 291 MB; both regenerate deterministically by rerunning
the repair, whose code hash is recorded in the log.

```powershell
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_reflection_repair.py --shot music_videos/reporter_test/shots/reporter_a --name r32_repair --start 0 --count 150
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/audit_r32_repair.py --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r32_repair/repair --compare r32_repair/source --label "R32 REPAIR" --compare-label SOURCE --fps shot --crop 848 204 265 178 --zoom 4 --out r32_repair_eyes.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r32_repair/repair --compare r32_repair/source --label "R32 REPAIR" --compare-label SOURCE --fps 12 --crop 848 204 265 178 --zoom 4 --out r32_repair_eyes_slow.mp4
```

Reports: `logs/r32_repair.json`, `logs/r32_withheld.json`, `logs/r32_flicker.json`.
Nothing from this round is committed. The r17 and r29 work sitting untracked in the
tree is unrelated and was not modified.

## 6. Starter prompt for round 33

> Stay on eyes. Read round32.md, then round30.md and docs/r30-eye-findings.md.
> **The artist has asked to return to the round-30 line of work** — coupled source
> geometry / light / material inference — rather than continue round 32's approach.
> Round 32's same-frame anatomical completion is **rejected**: it removes the
> highlight, beats round 31's flat-gain fill by 27–42 % on 12,082 withheld pixels,
> and still flickers, because every term of the fill is refit per frame from a few
> dozen pixels. Its colour traverses R 36→132 and B 2→66 across the shot with an
> adjacent-frame p95 of 12–16 and max of 52–59 display levels. Do not retry a
> per-frame-refit fill in any form; the finding is that no such fill can be
> temporally stable at a 24-pixel iris. What a next attempt needs is the eye's
> appearance estimated ONCE over the sequence in a stabilized frame, with only a
> smooth explicitly-constrained per-frame illumination term.
> Carry forward these measured facts so they are not re-derived: the highlight is
> **unclipped** (peak display 0.7843–0.8823, zero pixels at 255); the black beside it
> is undershoot **clipped at legal black by the tagged limited-range expansion**
> (green and blue exactly 0, red 8–26) and must not be replaced, because the artist
> reads it as the pupil; a pupil does **not** localize once highlight and rim are
> excluded (centre scatters over ±4 px on 6–10 pixels of evidence); and any radial
> eye model must stop inside the limbus, because the sclera is not radially symmetric
> about the iris centre and including it makes 106–197 of ~460 pixels look anomalous.
> Round 30's own remaining work is unchanged and still the target: training-only
> source-light initialization with enough directional reach, escape from inactive
> lobes, competition against spatial backing, credible 3D lid/light visibility, and
> actual scene-light replacement. Sources 805 and 815 are seen.
> **Working method the artist asked for explicitly:** put frames in front of him
> constantly rather than at the end, and inspect every frame on contact sheets at eye
> scale (the `composite/r29_shape/contact_*.jpg` standard) before claiming anything —
> a four-frame sample hid a failure mode for most of round 32. Report p95 and max,
> never a median alone, for anything temporal. Preserve exact keyed fallback, the
> 300/304 protected checks, generic input support and observed identity. No key
> retuning, no capture requirements, no impossibility claims. Produce a temporal
> review and round33.md with measured findings and remaining physical work.
