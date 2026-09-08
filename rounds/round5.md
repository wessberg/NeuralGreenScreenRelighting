# Round 5 handoff — round 4 finished: the dots are gone, the despill gate was wrong, and one metric disagrees with two others

**Date:** 2026-09-08. **Supersedes `round4.md`**, which is Codex's honest in-progress record of the
first half of round 4 and still worth reading for the edge-repair experiment and the eye plan.
Round 4 was started by Codex, which ran out of ChatGPT Plus credits mid-round; the artist handed
the remainder to Claude, which finished it. Nothing here claims photorealism achieved.

---

## 1. What was delivered, and the number behind each claim

### The dotted silhouette: two independent bugs, not one

This is why it had resisted a fix. Either one alone reads as "mostly better".

1. **Relief died at the matte edge.** `make_proxy_mesh.build` grows geometry past the matte on
   purpose, but the normal-integrated relief added to it is high-passed *through* the matte, so it
   fell to zero exactly at the silhouette — a one-cell ridge along the whole outline whose facets
   catch light. Found by the reviewer Codex ran; the nearest-extension fix is theirs.
2. **The relief was decimated without a low-pass.** It is high-passed at ≈ 40 depth px, so it holds
   energy down to 1 px, and the mesh samples every 2 px. Sub-Nyquist energy folds onto the grid
   pitch. Found in this half of the round; it is what fix 1 could not reach.

Measured by `scripts/measure_silhouette_corrugation.py` — RMS band-passed irradiance at the 5-plate-px
grid pitch, relative to local level, in a rim band, with the matte core as control:

| frame | rim, before → after | reduction | rim ÷ core, before → after |
|---|---|---|---|
| 0 | 0.181 → 0.052 | −71 % | 5.0× → 1.47× |
| 51 | 0.297 → 0.112 | −62 % | 3.2× → 1.24× |
| 79 | 0.242 → 0.064 | −74 % | 3.9× → 1.05× |

The core control moves by under 2 % throughout, so this is not a blur flattering the number.
Side effect worth knowing: mean dynamic range rose from **4.50 to 5.19 stops** — the terminators
got deeper, which matters for §3.

**Two hypotheses were tested and rejected. Do not re-run them.**
* Not render noise: 4× the samples (128 → 512) moves frame 51's rim figure by 4 %.
* Not the internal depth cliffs where the hand crosses the face: at frame 51 those carry *less*
  corrugation (0.087) than flat interior (0.096).

### The despill gate was wrong, and the artist found it by looking

`key_plate.py` gated despill on `excess = G − max(R,B)`, so it only fired where green beat **red**.
Her hair is dark brown (R .141 G .113 B .070): green sat 8 % above (R+B)/2 and read plainly olive
while never approaching red. Around her ear, **0.0 %** of subject pixels tripped that gate and
**11.8 %** were green-leaning. The despill was doing nothing at all on the exact pixels he pointed at.

Fixed as a shot-level `despill_limit`, defaulting to `max` so every existing shot — including the
proof-of-concept plate with its green costume — is untouched. `reporter_a` is set to `average`,
the standard green-limit gate: it corrects 8.3 % of subject pixels for a mean green change of
−0.46 %, and cannot overshoot, because a neutral pixel has `G == (R+B)/2` and is a fixed point and
the correction can never drive green past that same average.

**Read this part twice.** Three separate measurements said the plate was clean — 0.0 % green-positive
in that box, residual excess under +0.009 frame-wide, no cyan residue. All three measured
`G − max(R,B)`, *which is the buggy gate itself*. The metric inherited the bug's assumption and
therefore confirmed the bug. His eye was the only non-circular instrument in the room. On edge
integration and anything hue-related, look before defending a number.

### New pipeline stages and validators

| script | what it is for |
|---|---|
| `validate_proxy_detail_continuity.py` | depth step across the matte boundary, with the *interior* control the original table lacked |
| `measure_silhouette_corrugation.py` | the dots as a physical quantity at the grid pitch, plus labelled before/after panels |
| `measure_temporal_stability.py` | deviation from the 3-frame temporal median, split into core and rim |
| `review_video.py` (**stage 12c**, in `AGENTS.md`) | the review encode, so no round re-derives it by ffprobing the last one |

`review_video.py` exists because rounds 1–4 each rebuilt the same ffmpeg line by probing the
previous round's file. **A round is not shown until there is a video**; frames on disk are not
something he can judge.

### The 150-frame rebuild

**The fixed data was PROMOTED into the production paths at the end of the round**, so there is no
stale-default trap left: `relight/proxy/` holds the 150 fixed meshes, `composite/ratio_render/` the
128 spp render made from them, and `composite/ratio/` the despilled composite that is the
deliverable. Round 3's versions of all three were deleted — the artist confirmed he does not want
round 3 reproduced, and it is reproducible from code anyway via `make_proxy_mesh.py
--no-detail-extension --no-detail-antialias` plus `despill_limit: max`.

Videos in `final/`: `r4_round3_vs_round4.mp4` (whole round, before on the left),
`r4_despill_before_after.mp4` (despill isolated), `r4_despill_hair_1to1.mp4` (despill at 1:1 on the
hair crown), `r4_geometry_before_after.mp4` (geometry isolated). **These are now the only record of
the round-3 look.**

**The re-key left every matte bit-identical** (md5 on frames 0/51/79/149), so the meshes and the
render stay valid — despill only rewrites `fg`. If you ever re-key for another reason, check this
again before assuming the geometry survives.

---

## 2. Reviewer claim that needed correcting, not discarding

The reviewer reported "every front vertex inside the matte unchanged bit-for-bit". That is exactly
true against the matte `build` itself uses (alpha resized to the 768×432 depth grid): **0.000000 mm**.
It is false against the full-resolution matte, where 8 of 25 578 vertices move up to 65 mm — all of
them at erosion depth 0.00 px.

That gap is a finding rather than a rounding detail, and it is now logged: **geometry decides
"inside" at 768×432 while the composite decides at 1920×1080**, so a band about 2.5 plate pixels
wide around the entire silhouette is extrapolated geometry multiplied into photographed pixels.
That is the same band the dots lived in. Nobody has looked at what else lives there.

---

## 3. The open contradiction — do not resolve it by picking your favourite

`measure_temporal_stability.py` on the final sequence versus round 3:

* matte core **−11.7 %** (mean 0.293 → 0.258) — stabler, as hoped
* silhouette rim **+20.3 %** (mean 1.376 → 1.656, p95 3.18 → 5.42) — *worse*

But measured on the geometry directly, over 15 consecutive frames, RMS deviation of front-vertex
depth from its own 3-frame temporal median, in the rim band:

| build | rim | core |
|---|---|---|
| neither fix | 1.788 mm | 0.338 mm |
| extension only | 1.037 mm | 0.338 mm |
| both fixes | **1.023 mm** | 0.334 mm |

**The mesh got 43 % more temporally stable, not less.** And the artist watched the sequence and
said it looked better and temporally stable enough.

So two independent readings say improvement and one composite-level metric says regression. The
likely explanation is that the composite metric normalises deviation by the *local level*, and this
round deliberately deepened the shadows (+0.69 stops mean dynamic range): where the denominator
shrinks, relative deviation inflates, so a correctly darker terminator scores as less stable.
**This is a suspected metric artefact, not a confirmed one.** It is not resolved and must not be
written up as either a regression or a win. The clean test is an absolute-difference variant of
the same measure, gated on regions whose level did not change between the two sequences.

---

## 3b. The ceiling the artist named, and it is architectural

Watching frame 21 he said: *"I can kind of see the shine in her hair from the original green
screen room where they recorded her."* He is right, and it follows from the architecture rather
than from any setting.

`out = plate x R` with a smooth R **cannot move, reshape or remove a specular highlight** - it can
only recolour one. So every specular in the plate keeps encoding where the original softbox was:
the sheen band across her crown, the glints on her lower lip, the highlight down her nose. The
pipeline ADDS new specular (the additive G term, computed from the neural normals) but it never
REMOVES the old. Light the shot from the side and her hair still shines where a green-screen room
lit it from above.

This is the specular analogue of what `make_albedo.py` does for diffuse: divide out `E_source`
before re-lighting, or the plate is lit twice. Nothing does that for the specular layer. Candidate
approach is a specular/diffuse separation on the plate - tractable here because the highlight is
near-neutral while hair and skin are chromatic - then re-adding a specular lobe driven by the new
lighting. It is real work and it is not a tuning knob.

Until it exists, **constraint 5 has a ceiling below the bar**, and the shots where it will show
worst are exactly the ones with a hard key from a direction different from the original room's.

His full verdict on frame 21, worth keeping verbatim: *"it looks just a little bit
green-screeny/unnatural/unmatched lighting. It's the way the hair is shining and how her lips
look, and the lack of pores in the skin. It's good, sometimes great even."* The hair shine is
explained above. The lips and the pores are not yet explained - see 4.3.

Also his verdict on the first despill A/B: *"I can't really tell... it's always lower resolution
when they're side-by-side."* `review_video.py` now takes `--crop X Y W H --zoom N`. Use it. A
full-frame side-by-side is 3840 wide and every player scales it to fit, so judge fine work on a
1:1 crop of the region in question, never on a shrunken whole frame.

## 4. Genuinely open, in rough priority order

1. **Catchlights — not started at all.** The artist's original ask for round 4, deferred by him to
   round 5. `face_track.py` already computes 478 landmarks and throws away iris centres 468/473
   after using them for IPD. Round 3 and round 4 §4 both carry the plan; the constraints are that
   catchlights must be tracked, occluded by blink/eyelid, driven by the actual scene lights, added
   **separately and additively**, and must never contaminate the irradiance ratio or replace a
   photographed pixel. No painted fixed dots.
2. **Frame 51's interior floor.** Its core corrugation is 0.092 against 0.036 and 0.062 for frames
   0 and 79. That is real 5-px structure across her face, and it is neither render noise nor
   aliasing — both were ruled out. It only shows under 51's hard dim light. This is the strongest
   available lead into his "uncanny, too smooth" note on 51, and nobody has followed it.
3. **"Lips look wrong, skin has no pores" - unexplained.** The obvious suspect was the additive
   specular diluting photographed micro-contrast, since it adds smooth light carrying no texture.
   **That test was run and is INCONCLUSIVE, not negative:** at frame 21 the specular term barely
   fires at all (specular 1.0 vs 0 differ by max 4/255, 0.00% of pixels changed by more than 1),
   so that frame cannot answer the question. Re-run it where specular actually contributes - a
   close hard key. Also live: rows 9 (MTF) and 10 (grain) are both red and untouched, and a
   composite with no grain at all is missing one of the strongest cues of photographic origin.
4. **The residual ridge.** The crossing step fell 34.2 → 6.6 mm, against a 1.56 mm interior
   control, so the ridge is reduced rather than gone. Codex's suggestion — extend the final `Yw`
   rather than only `Yd` — is untested. Note some crossing step is physically correct: a real
   surface turns away sharply at a silhouette.
5. **Products derived from the PRE-despill plate are now stale.** `relight/albedo/`,
   `relight/skin_mask/`, `relight/normals/` and the measured `E_source` in
   `logs/source_light.json` were all computed from the old `fg`, and the delivered composite
   used that stale `skin_mask` and `E_source`. The despill changed green by a mean of
   −0.46 % on 8.3 % of subject pixels, so the error is small — but it is real and nobody has
   quantified it. Regenerate them before any round that leans on albedo or on the
   source-light fit, and re-measure rather than assuming the change is negligible.
6. **The 768×432 / 1920×1080 matte mismatch** from §2.
7. Constraint rows 3, 9, 10, 11, 12, 18 remain 🔴 and untouched by this round.

---

## 5. Settled — do not re-litigate

* **The ratio composite is the delivery architecture.** Never render the performer and show it.
* **The image-space edge repair is not the fix.** `composite_ratio.py --extend-edges` still exists
  as a control and is **off by default** — Codex left it on by default, which would have shipped it
  silently. Radius 8 leaves dots; radius 24 flattens real rim light. The fix was the mesh.
* **Removing the proxy's side walls is not the solution** either; it opens the SSS/shadow volume.
* `use_persistent_data` stays off. Samples are nearly free. The PoC shot's key thresholds are
  hand-set and stay that way.

## 6. Housekeeping — done, and what is left

Cleared at the end of the round: the `key/*_predespill` backups, every superseded round-4 test
directory, the stress-proxy and compare frame dumps whose videos survive in `final/`, the stale
`relight/proxy_render/`, and the orphaned `.claude/worktrees/round4-completion` worktree left by
the background Claude session Codex launched that never ran. `reporter_a` went from **4.5 GB to
3.1 GB**. Kept deliberately: `composite/r4_compare_final/` and `composite/r4_spill/`, the panels
behind the corrugation and despill findings.

`logs/proxy_mesh.json` was re-synced to the promoted meshes; it had been describing the round-3
build, because a `--output-dir` run writes its summary beside the meshes instead of into `logs/`.

Available if space is ever wanted: `music_videos/proof_of_concept/shots/night_listening_room/`
`intermediate/` is **6.5 GB** of regenerable linear 4K EXRs for an already-delivered shot. Its
`master/` and `final/` are separate and must not be touched — that master is the validated
evidence for constraint 17.

## 7. Starter prompt for round 5

> Round 5: tracked catchlights. Read rounds/round5.md, then AGENTS.md. `face_track.py` computes
> iris centres 468/473 and discards them — store them, with iris contours, eyelid polygons,
> blink/visibility gates, coordinate metadata and an explicit missing-detection state, without
> disturbing the existing IPD scale consumers; it needs the facetrack venv. Then add corneal
> reflections driven by the actual scene lights, tracked, occluded by eyelid and by the proxy,
> added additively in the composite and never mixed into the irradiance ratio. Verify against
> controlled comparisons and the photographed catchlights already in the plate; no painted fixed
> dots, and no generative reconstruction of the eye. Then, if there is room, take the frame-51
> interior-corrugation lead in §4.2, which is the live thread into the "too smooth" verdict.
> Measure before and after, produce a review video with `review_video.py --compare`, and write
> rounds/round6.md. State ceilings plainly and do not report a constraint row as green while a row
> it depends on is red.
