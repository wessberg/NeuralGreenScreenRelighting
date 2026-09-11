# AGENTS.md — working guide for any coding agent on this project

Canonical, provider-agnostic. `CLAUDE.md` imports this file; Codex CLI and other agents read it
directly. Put project knowledge HERE, never in a provider-specific file, so a fresh session on
any model starts from the same state.

## Read the latest round first

**Artist steering after round 30 (2026-09-09): selective generation is allowed.**
The artist explicitly accepts generative repair for selected regions, especially
after prolonged failure of an observed-pixel method, and reiterates a preference
for DiffusionRenderer. This supersedes older blanket bans on generated performer
texture and the strict pixel-provenance requirement below. Preserve recognizable
identity, gaze, expression and timing; evaluate selective eye regeneration with
scene-light conditioning in a source/result video. Preserve original pixels outside
the selected repair region and retain source fallback for failed repairs. Do not
spend another round tuning the same optical solver merely because it is the inherited
starter prompt: round 30 is not close to a working eye solution. Existing tracking,
masking, lighting and compositor work remains useful. A model's generated output
must be judged visually and temporally; exact material recovery is not a prerequisite
for this newly authorized experiment.

**Round 31 artist rejection:** released SVD DiffusionRenderer now runs locally,
but whole-eye replacement produces unrealistic golden irises and distorted pupil
detail. Restricting its generated albedo to reflection spots still leaves a light
brown spot inside the pupil; the artist rejected that too. Neither is a repair.
Selective generation remains authorized; this does not reinstate a blanket ban.
Do not reuse the failed albedo as clean pupil material or hide its errors with
another tint/mask adjustment. See `rounds/round31.md` for outputs and next steps.

**Round 32 rejection:** same-frame anatomical completion of the highlight was built
and **rejected**, for colour AND shape flicker and a sometimes-brown pupil, because
every term of the fill was refit per frame from a few dozen pixels. **No per-frame-refit
fill can be temporally stable at a 24-pixel iris.** See `rounds/round32.md`.

**Round 42 — the WET EYE is ACCEPTED: *"Yeah, it's a huge improvement! Looks super good."* `r42_wet` is the eye baseline. The bulb is a sub-pixel speck because it is 4 cm at 0.5–2.4 m (1–5°); the softbox subtended ~35° — a catchlight's size is the source's angular size, so a big soft reflection needs a big source in the scene, not a setting. Read `rounds/round42.md` §1, §4, §5,
§6 first.** The eye is now ONE surface of revolution per eye (`stress_proxy.py --corneal --globe
--socket`): the 7.8 mm cornea, a 1 mm concave limbal fillet, the 12 mm scleral globe (tear film
over conjunctiva, `--sclera-roughness` ASSUMED 0.08 — measured: the capture room's softbox
reflects in the globe UNDER the cornea on 100 % of open eyes, so the plate fits nothing here),
inside a black head-attached SOCKET cup (mediapipe head pose, visual-field cuts nasal 60 /
temporal 100 / up 60 / down 75°, 24 mm — 14 mm let the behind-rim bulb slip past). Without the
socket the globe reflected the fill panel from BEHIND her head onto the sclera. The old room's
glints on the white of her eye are removed on top of the accepted iris run
(`sclera_glint_removal.py`, footprint held in time, fill from the surrounding sclera, flicker
equal to the untouched sclera's); the corner tear-lake glint is left (`--corners`). Composite:
`composite_ratio.py --eye-removal r42_wet_removal --corneal-dir r42_eye_pass --corneal-blur
0.35 --corneal-matte lids` → `r42_wet`. Validated: bulb in the globe vs the nearest rendered
peak 0.04 px median (p95 0.15) on the 25 socket-open eyes, wrong conventions 6–20 px, cap
unchanged at 0.23 px. **Not built, and measured why:** a tear line (the plate's lower-lid band
is ≤ 0.09 display above the sclera; a meniscus model predicts glints 9–19 px from where the
band is brightest). **Physics he may reject on sight:** a source 60–80° off axis reflects TWICE
at the limbus (cornea and globe, 5 px apart — the fillet cannot bridge it, a real sulcus does
not either); the 60 cm fill at 0.6 m images as a flat clipped lobe on the sclera at camera
left/right and as a second grey square beside the corneal one at 127. Videos
`reporter_a/final/r42_wet_compare.mp4`, `r42_wet_eyes.mp4`, `r42_wet_panel.mp4`.

**Round 41 — the softbox removal is ACCEPTED (`r40_floorB`), and the tour's bulb now reflects
in her corneas on top of it. Read `rounds/round41.md` §4 and §6 first.** The artist: *"I'm
pretty satisfied with our removal of the softbox reflection from the eyes now"*, then *"make
sure the lightbulb reflects in her eyes correctly"* on the bulb-tour scene. Built:
`stress_proxy.py --corneal` (round 34's dielectric corneal cap lit by the tour itself, bulb and
fill visible to reflection rays only, no proxy occluder), anchored on the fitted iris centre
behind the measured gaze (`corneal_anchor.anchor_limbus`; the plate's own softbox glint grades
that at 1.31 px rms against 1.87 for a camera-axis gaze, `validate_tour_catchlight.py
--plate-glint`), and `composite_ratio.py --eye-removal <run> --corneal-dir <pass>
--corneal-blur 0.35` (the accepted removal substituted under the ratio; the pass ADDED through
the fitted lids and the limbus in the ratio's own anchor units, softened to the plate's PSF
measured on its glint by `measure_glint_psf.py`). Render vs an independent numpy reflection:
0.20 px median, wrong conventions 1.3–5.5 px. Videos `reporter_a/final/r41_tour_compare.mp4`,
`r41_tour_eyes.mp4`, `r41_tour_panel.mp4`. **Verdict:** *"looks decent enough. Maybe slightly
unrealistic? Maybe the eye didn't become 'wet' enough?"* — the wet look is round 42's job. Two things to ask
about first: the 60 cm fill panel reflects as a soft grey square when it is in front (behind
rim, frame 127) and as a pink crescent CUT AT THE LIMBUS at camera left/right (52–85) — a real
eye carries it onto the sclera and hides part of it behind the nose; neither is built. The
bulb itself is a 2–3 px spot that clips in the 8-bit review (physics; the master keeps it).
`import_overlap_eyes.py` carries `reporter_full`'s accepted run and tracks onto `reporter_a`
(same plate, offset 700, gated on fg sha256) — it exists only because removal and tour live on
two shots of one plate. Round 40's ellipse fit is superseded by this work, not abandoned.

**Round 40 — 933–955 right was the LOWER LID, not foreshortening; the real off-axis class is
the hard sideways GLANCE, located but not built. Read `rounds/round40.md` §2, §3, §6 first.**
76 % of every skin pixel the ungated fill paints sits below the limbus centre at rho
0.62–0.72: `eyelid_fit`'s lower walk takes only bright-to-dark steps, so where her lid covers
the iris (dark-to-bright) it sampled nothing and extrapolated the sclera columns 3–5 px low.
`eyelid_fit.py --lower-iris-edge` adds a separate `lower_iris` fit (the lid over her iris,
relative to HER iris/skin levels; `lower` untouched, GPU = CPU exactly) and
`eye_removal_gpu.py --lid-floor` bounds the fill by it — per-column median of 3 frames, a
missing floor = no bound, each fit held to its own column span. Run `r40_floorB`: 15 right eyes
back from softbox-only to the full fill, nothing else changes status, 758 frames identical,
canonical identical. **Artist verdict not yet recorded.** Folding the iris edge into the
lower-lid parabola was rejected: the lid inset reads **max(upper, lower) rms** and couples
the lids, and it un-dropped 1050 left onto a displaced circle (round 39 §2a's second pupil).
At a hard glance (961–964, 1050–1054, 125 left; the head turn 110–214) her iris is
foreshortened and the fixed-radius circle sits beside it; `probe_r40_ellipse.py` separates
those (+23–37 % over the best circle at b/r ≈ 0.63; ordinary eyes 0 %) except beside a glance
(+3–8 %, ambiguous). The pass already paints ax ≠ ay; **round 41 builds the ellipse fit**,
per the artist: *"that should be part of a fresh round in a fresh session"*.

**Round 39 — the eye stage runs on the GPU and reproduces the CPU BIT FOR BIT; round 38's
evidence frames are fixed at their causes. Read `rounds/round39.md` §2, §2a and §6 first.**
`eye_removal_gpu.py` is the eye pass now (21 min → ~64 s per 1200 frames; every PNG of
`r38_full_own` identical), `limbus_fit.py` / `eyelid_fit.py` default to `--device cuda`
(tracks identical, 3 min / 67 s → 72 s / 9 s). Iterate on the GPU scripts; the CPU ones are
the reference. Exactness took cv2's and numpy's real arithmetic, pinned in `eye_gpu.py
--selftest` (round39.md §3) — keep it green when touching a primitive, and gate a port on
`validate_eye_gpu.py`. The fixes (all flags): `--lid-claim 0.8` (no lash line + a lid
claiming the crease = softbox only: 137, 304 L), `limbus_trust.py` + `eyelid_fit.py --update`
(bridge a slid limbus run from trusted frames: 243 L), `--skin-gate 6 --skin-gate-low 3`.
Painted-skin eyes 49 → 0 (`audit_r39_classes.py`; look with `review_r39_classes.py`).
**The first video sent (`r39_fixE`) had a regression I introduced: a bridge across a fast
glance painted a displaced iris at 1049–1055 L.** A high limbus SE is motion blur as often
as a slide; only the image can say which answer is wrong, so `limbus_trust.py` now keeps a
bridge only where the fit's own edge objective prefers it, per bridged run (§2a). Corrected
run `r39_fixF` — **artist: *"looks good"*; it is the baseline for round 40.** **Look at
every eye a new stage changes, against the previous run, before sending a video** — the
audit's flicker max pointed at it and no class caught it.

**Round 38 — the whole 20 s clip (`reporter_full`, 1200 frames): "really, really close to
perfect overall", with failures "here and there across the full 1:40". Read
`rounds/round38.md` §2 and §6 and `rounds/evidence/round38/evidence.json` FIRST.** The
artist's evidence frames are 137 (whole disc painted over a squint: the upper bound sits on
the brow crease), 243 (double iris: a SEVEN-frame limbus slide the one-frame gate cannot
fix) and 304 (iris over the lid: a lash line that cut nothing), all on her LEFT eye. Round 39
**moves the eye pass to the GPU first** (it ran at 1 % GPU, single-threaded numpy, 21 min per
1200 frames), gated on reproducing `reporter_full/composite/r38_full_own`, then fixes those
classes over all 2400 eyes. Also logged, not in scope yet: a second capture-room glint on the
white of the eye (`audit_r38_sclera.py`). Per-shot constants that do not scale with clip
length were found and replaced by flags: `--canonical-votes-frac`, `eyelid_fit.py
--curvature-window`; `--fill-residual-gate 8` catches a lid fit on the crease at a blink edge.
`--canonical-from` is DIAGNOSTIC ONLY — the tool is generic and may not borrow another
run's iris.

**Round 37 — accepted "based on just the presented frames" (source 700–849, 150 of the
clip's 1200). Round 38 runs the whole 20 s clip. Read `rounds/round37.md` §0, §5, §6.**
The command is in round37.md §7; every piece is a flag and defaults are unchanged. What it
established, none of which should be re-derived:

* **The softbox is replaced whole by GEOMETRY** — its blob plus a 2 px disc
  (`--specular-reach`), after every guard. The per-column lash ceiling dropped to the lower
  iris in every column the softbox crosses; the photometric exemptions that patched the
  notch left its falloff standing as the grey OUTLINE the artist saw. The lash line is now a
  shot-curvature parabola through the trustworthy walk samples (`--lid-line parabola`).
  **Do not bring back `--lash-bright/ring/halo`.**
* **The canonical iris's top half was pooled from 6–19 frames** (the band below the eye
  centre gets ~128). It carried a dark band through the pupil, lid-skin patches and lash
  streaks into every frame. `--canonical-votes 60` rebuilds poorly seen cells from their
  mirror across the horizontal diameter. **Look at `composite/<run>/canonical_<side>.png`
  whenever the atlas changes.**
* **Shading comes from HER pixels at the fill's edges** (`--top-shading`, `--region-rows`).
  It is never read off the reflection or anything brighter than 0.45 display, and is
  **filtered in time** (`--shading-temporal`): a 5-frame median in the eye's own coordinates,
  SYMMETRIC windows only (one-sided lags at a blink), and only frames that PAINTED a spot
  vote on it. Unfiltered, it was the "flicker … small color variation here and there" the
  artist found in R37h and **no static tool could see**. Use `audit_r37_flicker.py` and
  `review_r37_temporal.py`: stabilised, CIELAB, global vs local.
* **At a blink the nearest good frame is always the more OPEN one.** Never copy a lid or a
  lash line from one side there; it paints iris onto a closing lid (704/705 left). A frame
  with no trustworthy upper bound gets only the softbox's own region.
* **The limbus fit slides on single frames under a low lid** (3.7 px at 717 right).
  `gate_limbus()` catches it against the eye corners, but **only when the limbus itself
  breaks from its neighbours AND the corners do not**. The corners jitter too (764–770 left).
* **Any measurement that rides the limbus track inherits its slides.** R37r's worst
  "flicker" was exactly where the gate had fixed the track. Check in corner-stabilised crops.
* Tools that found things, in `scripts/`: `review_r37_sheets.py` (stacked tight crops of
  every frame), `audit_r37_eyes.py` (fragments, lid shadow, lash, **region standout**,
  flicker, departure), `audit_r37_flicker.py`, `review_r37_temporal.py`,
  `review_r37_panel.py` (face scale).

**Round 36 was REJECTED, and its most important finding is about the AGENT, not the code.
Read `rounds/round36.md` §1 first.** The artist's closing words were *"Apparently you can't
see it. Hand off the round, I'll tackle it fresh in the next one."* Round 36 repeatedly
declared frames clean that he could see were not — at 18x, after being told to look, twice
about its own fixes. **Treat round 36's visual assessments as unverified; its numbers are
reproducible and its adjectives are not.** Put the previous result and the new one in the
SAME tile, one above the other, at high zoom, and step every frame: differences that are
invisible in isolation are obvious in comparison, and that is what finally exposed the
defect he was pointing at.

**Lash, lid-shadowed iris and the specular's falloff OVERLAP IN LEVEL on this plate, so no
photometric lash line can work.** At source 740 left her lash and her shadowed upper iris
are ONE continuous dark run at 0.16-0.22; lashes are 0.13-0.35, shadowed iris 0.15-0.30,
the specular's falloff 0.30-0.45. Six guard variants are tabulated in round36.md §0 with
what each cost, and every one trades "don't paint my lashes" against "the softbox is still
there". **Do not build a seventh.** The way out is a geometrically correct lash line:
`eyelid_fit.py` is audited to 0.43 px and still sits **seven pixels high on the LID CREASE**
at source 727 left and **abstains entirely** at 723 left. Round 35 §3 believed it had fixed
the crease failure; it had not.

**The LID SHADOW model is DEFAULT OFF, rejected on sight.** Fitted as
`1 + A·exp(-depth/tau)` per eye (A = +0.44 left), it put a light washed band across the top
of the iris and **erased the lid shadow the artist explicitly asked to keep**, and the
bright band above the pupil made the pupil read as displaced. The pooled depth profile
behind it is real; applying it as a per-eye constant over-brightens every frame that was
already right. **Validate a correction like this at FACE SCALE on frames — a pooled profile
cannot see a halo.**

**A metric must be built from what the EYE does, not from what the repair believes.** The
atlas's softbox test located the specular in the SOURCE by a threshold and checked those
pixels in the result, so when the guard sheared off the highlight's corner it reported
**zero eyes worse on every frame** while the artist could see a white block. It now reads
the RESULT alone — a bright blob against a smooth iris — and reproduced him at once.

**`ncc = 0.0` out of `atl.build` is an initialiser, not a correlation.** `register()`
returns None when it finds no overlap; source 843, 846 and 848 left carried exactly 0.0
between neighbours at 0.985/0.988/0.975, were declined outright, and kept the whole softbox
— the *"glimpses in the last seconds"*. Frozen registration places every frame by the same
median offset, so an unregistered frame is no longer vetoed.

**The limbus error is VERTICAL and the fit does not know it is wrong.** Median |offset_x|
0.05 px against |offset_y| 0.42-0.50, excursions to −4.94 px, every one on a frame with
15-24 edge points instead of 41: with the radius fixed, a side-only arc says almost nothing
about the centre's HEIGHT, so the circle slides down and snaps back. `--trust-se` shrinks a
frame toward its neighbours by its own reported standard error and **barely fires (6 and 21
frames)**. Same failure mode as the eyelid fit on the crease; likely the same fix.

**Anything refit per frame moves.** The lash ceiling, read per frame, jumped up to **0.8-0.9
iris radii (10 px) between adjacent frames**; it is now stabilised by a three-frame median
in the eye's own coordinates, which adds no lag on smooth motion.

**Round 35 — all three eye anchors now fit the IMAGE; the remaining seam is the LID
SHADOW, not the mask.** The limbus (`limbus_fit.py`) and both eyelid margins
(`eyelid_fit.py`) are fitted to the photograph and audited by estimators that share
nothing with them; the atlas registration, the last per-frame refit in the eye path, is
frozen to one offset per eye. Measured against her own image, the mediapipe iris ellipse
sat **3.08 px** from her iris centre and the mediapipe eyelid contour **8.34 px inside**
the true opening on the left eye. Both are replaced. **Do not put a mediapipe contour back
and do not re-derive these anchors.**

What the artist still rejects is the upper eyelid and lashes, and round 35 measured why:
**her iris darkens toward the lash by +0.082 display across the fill boundary and the
repair flattens it to −0.004**, because the canonical iris is painted with ONE scalar gain
per frame. That is a systematic step at the lash line on essentially every treated frame,
and **no adjustment to the eyelid mask removes it — moving the boundary only moves where
the step lands.** Round 36 should model the lid shadow against depth below the fitted lid
margin. See `rounds/round35.md` §7.

Three other things round 35 established, all of which cost a cycle first: **the canonical
iris must not be pooled through the same mask it is painted through** (lid and lash got
into its upper cells and drew a warm arc that is nowhere in her photograph); **the
canonical eye's grid must reach past the limbus** (92.5 % of surviving softbox pixels sat
at rho > 0.95 where there was no material at all); and **`audit_r35_defects.py` is the
artist's acceptance test** — a brighter lid or any surviving softbox, over all 300 eyes,
read from the photograph and the result only. Run it before and after every change.

**Round 34 — the construction is settled and the artist endorsed it; the TRACKER is the
blocker.** The eye is now generated **once**, in the eye's own registered frame, and
rendered into every frame (`eye_soft_removal.py --fill-iris`); only a scalar illumination
gain varies per frame, so iris crawl is structurally impossible. His words: *"the new iris
generation approach here seems to be the right one"*. What now blocks it is measured:
the mediapipe four-point limbus ellipse sits at **0.936 of the observed iris radius on the
left eye**, with **1.66 px** of spread and **2.49 px** p95 frame-to-frame change against
the photograph (`audit_r34_track.py`). That is his "double iris" and his "delayed, slow to
react", and every fill since round 32 inherits it. **Round 33's 0.353 px figure measured
smoothness against its own trajectory, not accuracy against the image — never quote it as
accuracy.** Round 35 fits the limbus to the image edge and feeds it to everything.

**Correction to a standing rule.** The black beside the highlight is coding undershoot,
as recorded — and that is exactly why it must be removed WITH the highlight, not left.
It is the codec's ringing *around* the bright lobe; deleting one lobe of a two-lobed
artefact leaves a hard black crescent round a smooth fill, which is what the artist
rejected as "weird brown stuff" in rounds 32, 33 and 34's first pass.

**Model class matters more than model rank.** SVOR 1.3B and EffectErase 1.3B (both CVPR
2026) are OBJECT removal: given a mask on an iris they delete the iris — SVOR returned
blank sclera, EffectErase a lump of skin. **UnReflectAnything** is specular removal and
keeps her eye; **ProPainter** is flow propagation and invents nothing. The last two are
equal on temporal stability here (per-cell spread 20.7 vs 21.1). ProPainter needs the
occluder to MOVE between input frames: at stride 1 it puts the catchlight back, at stride
5 it removes it. All four are NONCOMMERCIAL. Do not re-evaluate the object erasers.

**The scene's own corneal catchlight is built and validated** (`render_eye_catchlight.py`,
`validate_eye_catchlight.py`: NCC 0.942 against an independent numpy model, three wrong
conventions at 0.077 / 0.007 / −0.012; 110,213 added pixels inside the eyelids and 0
outside). **The artist parked it until removal is clean — do not switch it on unprompted.**

See `rounds/round34.md`.

Also still true, and not to be re-derived: the highlight is **unclipped** (peak display
0.7843–0.8823, zero pixels at 255); the black beside it is undershoot **clipped at legal
black** by the tagged limited-range expansion (G and B exactly 0, R 8–26) — and round 34
established it must therefore be removed WITH the highlight, not left, because it is that
highlight's own ringing; and no pupil localizes once highlight and rim are excluded, which
is why the pupil is now DRAWN into the canonical iris at a level measured from her own
darkest observed pupil rather than recovered.
Two working rules he stated explicitly: **put frames in front of him constantly, not
at the end**, and **inspect every frame** on contact sheets at eye scale (the
`composite/r29_shape/contact_*.jpg` standard). Two of round 33's four bugs were
invisible in the logs and visible only in frames. Report p95 and max for anything
temporal, never a median alone, and never an adjacent-frame step alone — it conflates
real eye motion with flicker, which is how round 32 scored below its own control while
plainly flickering.

**Before doing anything else in this repo, read the most recent file in `rounds/`.**

```powershell
Get-ChildItem rounds/round*.md | Sort-Object { [int]($_.BaseName -replace '^round', '') } | Select-Object -Last 1
```

Each round is the handoff from the previous working session: what was delivered and verified,
what is genuinely broken, decisions the artist has already made (do not re-litigate them), and a
starter prompt for the round you are about to do. The latest round supersedes this file and the
README wherever they disagree.

At the end of a working session, write the next `rounds/roundN.md` in the same shape, ending
with a starter prompt for round N+1.

**Round 8 reflection findings:** the original catchlights, crown sheen and lip gloss are still
unresolved. Disabling added GGX changes lip pixels by only 0.00955 display levels on average;
it is not the lip fix. Simple centre/scale temporal iris recovery fails a withheld-pixel test
(median relative RGB error 12.47–14.50%, p95 50.10–64.59%). Unclipped reflections are unresolved
sums, not automatically recoverable iris texture. `audit_r8_reflections.py` is an isolated
reporter diagnostic, not a production repair or validated removal mask. See `rounds/round8.md`.

**Round 9 correspondence findings:** masked affine iris alignment improves withheld-pixel
error on matched samples (median relative RGB 11.67% → 7.27% left, 10.45% → 6.09% right),
but provides **zero confident pixels at the detected original catchlights**. The new test
excludes shared native interpolation footprints, uses identical controls, and reports
abstentions. `audit_r9_iris.py` is diagnostic only. Original catchlights, hair sheen and lip
gloss remain unresolved; production defaults are unchanged. See `rounds/round9.md`.

**Round 10 full-source findings:** expanding the iris donor pool from 150 to all 1,200
original frames still gives zero confident samples beneath detected reflections. An
exhaustive donor probe at fixed targets does not rescue coverage. Native-image inspection
finds iris/aperture regions admitting sclera/skin, lower-rim false candidates and missed
softboxes; `found=True` and zero detected glints do not establish clean iris observations.
All three original complaints remain unresolved. See `rounds/round10.md`.

**Round 11 artist decisions and scope:** this is an input-agnostic tool; the reporter's
failed donor search does not rule out temporal reconstruction on other footage or with
better correspondence. Reconstruct from observations of the same iris where reliable.
For unreliable iris detection/angles or correspondence, **keep the source keyed eye,
including any original catchlight**. This fallback is explicitly accepted by the artist.
`temporal_iris.py` implements a generic pupil/limbus coordinate sampler and exact source
fallback with a taper inside reliable runs. Its positive reconstruction evidence is
**synthetic with known geometry/masks only**; the real automatic detector/correspondence
and compositor integration are not built. All 150 real-frame fallback checks preserve
the keyed source exactly. Do not present this as a reporter reflection fix. See `rounds/round11.md`.

---

**Round 12 findings and steering:** `iris_observations.py --shot` now collects reusable
native keyed-eye candidates with dynamic crops, roll and pixel-aspect metadata; all
validation flags remain false and source fallback is exact. On real withheld pixels,
fitted torsion worsens paired median display error (3.2601 to 3.6259); zero pixels pass
the combined correspondence/agreement gates. This is not a reflection repair. The
artist reiterated generic input support, asked for existing neural solutions, and
clarified that **tracking plus scene-light-based replacement remains the full task**.
Evaluate pretrained eye segmentation/tracking before further custom fitting; see
`docs/iris-prior-art.md` and `rounds/round12.md`. No neural model was run this round.

**Round 14 neural correspondence:** a frozen donor/target split and fresh sparse
source-only annotations now exercise EllSeg/RITnet proposals through masked
correspondence. The fixed-hole test has 500 attempted native pixels, 57 paired
supported pixels, and **zero passing the combined context/agreement gates**.
Paired median display error worsens from 4.2344 to 4.3501 with fitted torsion.
These are candidate-region scores, not certified clean-iris accuracy. The candidate
gate also accepts the nearly closed left eye at source 984. All six sparse real
reflection probes have zero donor support; pupil crossing cannot be excluded.
Search-wide native-footprint exclusions and exact keyed fallback are verified.
No production repair or scene reflection was enabled. See `rounds/round14.md`.

**Round 15 dense correspondence:** released RAFT large now runs in the existing
isolated neural runtime. With six nearby donors, all 30 reused hidden iris points
have raw support; paired median display error improves from 5.8333 (fixed tracked
crops) to 2.6596 (RAFT). This is seen-data observed-pixel evidence, not clean-material
certification or a comparison with round 14's distant donors. Candidate support is
13/30 for RAFT, but all-arm context exclusions leave at most three context pixels,
so the eight-pixel gate is unevaluable and zero repairs are earned. All 12 real
reflection points have zero candidate donor support. At nearly closed source 981
left, 4/5 points pass round-trip consistency: this is not visibility. Hidden RGB,
native footprints and exact keyed fallback are verified. See `rounds/round15.md`.

**Round 17 reflection material:** all 12 reused reflection anchors now run one
query at a time through masked CoTracker3 and RAFT against the same eight labelled
donors. Of 96 attempts each, RAFT cycle accepts 27 (14 labelled reflections,
13 unknown); CoTracker accepts 15 (eight reflections, seven unknown). Neither
has an accepted iris-compatible footprint; the shared neural proposal rejects
all. Tracking a reflection is not evidence of tracking the underlying iris.
RGB-free preflight on 30 iris / 12 reflection queries finds only five of 126
target/donor pairs with capacity for eight context samples, and no query with
three eligible donors under the frozen fixed/+3px control design. No photometric
scores or repair are earned. This is a limitation of this design and labels,
not universal impossibility. All 24 complete-footprint substitution runs and
300 exact keyed fallbacks pass. See `rounds/round17.md`.

## What this project is

**Round 16 temporal visibility:** released CoTracker3 scaled offline rejects all
20 reused iris queries at source-labelled closure frames 981/982 (both eyes);
RAFT cycle alone accepts 5/20, while RAFT plus its proposal also rejects all 20.
New source-only region annotations are independent of predictions, not independent
human truth or certified clean radiance. Across 80 attempts, CoTracker marks 36
visible: seven footprints fit conservative iris interiors, 29 remain unknown.
At narrowed 985 left it accepts 2/5; those materials remain unresolved. Controlled
opaque-occlusion tests reject 128/128 hidden samples and recover 128/128 afterward;
these are synthetic geometry tests on observed RGB. Exact source fallback remains
verified. No reflection removal or scene replacement was enabled. See `rounds/round16.md`.

**Round 13 neural evaluation:** released EllSeg and RITnet weights now run in
`dependencies/ellseg-runtime/.venv` with torch 2.8/cu128. Reusable
`neural_iris_observations.py --shot` stores separate complete-ellipse and visible-part
proposals, retaining exact source on all 300 keyed eyes. On reused sparse labels,
EllSeg admits 40/40 iris and 16/60 noniris points; RITnet 30/40 and 0/60, but has a
small false pupil/iris in one blink and a large pupil-boundary miss on source 923.
These are not validated material masks, clean donors or repairs. No production
reflection change was enabled. Read `rounds/round13.md` and
`docs/neural-eye-evaluation.md` before extending this path.

A virtual-production pipeline that composites green-screen footage of the artist into Blender
environments, to produce music videos. The stated acceptance bar is **full photorealism** —
"if it doesn't look 100% authentic and like I was literally there, I don't want it anyway" — so
do not present incremental gains as success. Say plainly when an approach has a ceiling below
that bar.

The artist shoots his own performances at **24 fps**, usually in bust/medium framing with no
feet or floor contact in shot.

### His capture rig — `capture_format.json`

**Panasonic LUMIX S5 IIX, 6K 3:2 open gate 5952×3968, HEVC 10-bit 4:2:0 Long GOP 200 Mbps,
V-Log / V-Gamut · SIRUI Saturn 35 mm Full-Frame Anamorphic, 1.6× squeeze · 180° shutter at
24 fps · WB 5600 K · delivering 3840×1600 (2.40:1).**
Full detail and consequences are in `capture_format.json`; read it before writing any stage that
touches camera geometry or colour.

**Two things will bite, and both are silent.**

1. **V-Log is not linear and V-Gamut is not Rec.709.** Every stage here assumes scene-linear
   Rec.709 — keyer thresholds, depth and normal estimators, the SH irradiance ratio, the PQ
   grade. Feed V-Log in raw and all of them are wrong while still producing a picture: lifted
   midtones, crushed saturation, and every number in `rounds/` quietly invalid. Decode on
   ingest, before keying, and never apply a creative display LUT first — it is not invertible
   and throws away the highlight range the HDR grade exists for.
2. **The plate is anamorphically squeezed (1.6×) and Blender renders spherically.** Depth and
   normal estimators must see the *desqueezed* plate, or they infer a face 1.6× too narrow —
   plausible-looking and wrong. Defocus is an oval, not a circle. `scripts/anamorphic.py` holds
   every factor of 1.6; never retype one.

**A third that will bite the keyer: the plate is 4:2:0 Long GOP.** Chroma is stored at half
resolution (2976×1984) and a green-screen key is a *chroma* operation, so upsample chroma
properly before keying or every matte edge inherits a blocky staircase no threshold will fix.
Long GOP puts mosquito noise around exactly the high-contrast edges that matter — the silhouette
and every hair — so transcode to an all-intra intermediate **before** keying, or the matte will
crawl. Key at native 6K and downscale afterwards: with 2.48× linear oversampling the matte comes
out effectively supersampled, and hair edges get real fractional alpha instead of a hard
decision. Downscaling first throws that away.

The numbers all agree with no crop, which is worth noting because it is not a coincidence to be
re-derived each time: 5952/3968 = 1.5000 exactly, desqueezed by 1.6 that is 2.400:1, and the
delivery frame is 3840×1600 = 2.40:1. Shoot open gate — a 16:9 mode desqueezes to 2.84:1 and
forces a crop.

These values describe **his** footage only. `reporter_test/reporter_a` is online stock footage
shot by someone else at 59.94 fps, Rec.709, on an unknown spherical lens; its camera block in
`shot.json` is the authority there, and it must not inherit the rig above.

---

## Hard constraints — what "photorealistic" actually requires

The bar is not a feeling, it is this list. **A shot is not done until every row is green.**
Do not report a stage as working while a row it depends on is still red, and do not present a
row's partial improvement as the row being satisfied. If an approach cannot reach a row, say so
and name the approach that can — that is more useful than a better number on the rows it can
already do.

Status is for `reporter_test/reporter_a`, the test bed, as of the end of round 2.

| # | Constraint | Why it is non-negotiable | Status |
|---|---|---|---|
| 1 | **Self-shadowing** — nose onto cheek, brow onto eye, jaw onto neck | The single strongest cue that a light is real, small and close. A gain map *cannot* do this at any quality: shading is `dot(n,l)`, shadowing is occlusion. | 🟡 proxy casts real shadows, validated (shadow travels 14.5 % of face height between uplight and toplight); **not yet in the composite path** |
| 2 | **Received shadows** from scene geometry crossing the actor | A window mullion or a passing shape that stops at the actor's edge is an instant giveaway. | 🟡 the proxy is in the scene and occludes; needs an environment render rather than the test lights |
| 3 | **Cast shadows onto the environment** | Only matters where the actor is near a surface; bust framing usually hides it. Shadow Catcher pass. | 🔴 |
| 4 | **Subsurface scattering** in skin | Skin is not opaque. Without it, ears, nostrils and lit edges read as painted plastic. Light transport through a volume; no per-pixel multiply can fake it. | 🟡 `RANDOM_WALK_SKIN` + Jensen 2001 radii on the proxy, dedicated SSS passes enabled; not composited |
| 5 | **Specular response** — roughness, Fresnel, sheen on skin and fabric | A purely diffuse actor under a room full of practicals has no catchlights and no wet look. **And the ratio composite cannot remove the plate's OWN speculars**: `plate × R` with a smooth R recolours a highlight but cannot move or delete it, so the original room's sheen survives every relight — the artist spotted this on sight. Needs specular/diffuse separation on the plate, not a setting. | 🟡 **unchanged, and a documented dead end behind it.** Round 5 built a dichromatic separation, claimed it worked, and was rejected on sight: it removed 26.9 % of tooth energy and 52.7 % of sclera energy, blocked up dark hair, and painted reflections over eyelids. Reverted; defaults are round 4's. The original room's sheen, lip gloss and catchlights are all still in the photograph — see the rejection addendum in `rounds/round6.md` and `rounds/round7.md` |
| 6 | **Light direction, colour, level, falloff** | The baseline match. | 🟡 ratio composite delivers it; `E_source` now MEASURED from the plate rather than guessed (24 % relative RMS correction) |
| 7 | **Anamorphic lens match** — 1.6× squeeze, oval bokeh, correct FOV | The plate is anamorphic and Blender renders spherically. A silent mismatch here is geometric, not aesthetic. | 🟢 `scripts/anamorphic.py`: geometry exact, bokeh measured at 1.6:1 vertical (error 0.0). Blur *magnitude* still needs matching against a real defocused take |
| 8 | **Depth of field** matched to the taking lens | Nobody shoots a dim interior deep-focus; a sharp background is the "computer-ey" tell. | 🟡 f/2.0 in the environment render |
| 9 | **MTF / sharpness match** | The CG background is ~2× sharper than the plate. Measured blur on the background, not the actor. | 🔴 |
| 10 | **Grain match**, generated before quantisation | Also dithers 8-bit banding. σ ≈ 0.0042 sRGB, generated at 1080p and upscaled 2× because the plate is. | 🔴 |
| 11 | **Chroma subsampling match** (4:2:0 behaviour on the plate) | Sharp CG chroma against subsampled plate chroma separates the two layers. His own footage is 4:2:0 Long GOP, so this also governs the keyer: upsample chroma and de-Long-GOP before keying. | 🔴 |
| 12 | **Motion blur match** — 180° shutter at 24 fps | CG rendered without matching blur strobes against a correctly blurred plate. | 🔴 |
| 13 | **Edge integration** — no green fringe, correct edge alpha, no dark halo | Round 2 measured 68 % of hair-fringe pixels going green-positive once a green light was applied. | 🟡 despill works; fringe compounds under coloured light |
| 14 | **Temporal stability** — no flicker, no crawling normals or matte | A per-frame estimator with no temporal model will crawl. Sapiens2 measures 2.03°/frame. | 🟡 measured, not yet controlled |
| 15 | **Identity preservation** — preserve the actor's recognizable identity, gaze, expression and performance | Selective generative repair is explicitly allowed by the artist after round 30. Preserve original pixels outside the selected repair region; generated regions need visual and temporal evaluation. | 🟡 original ratio path preserves source; selective generation is authorized but not yet validated |
| 16 | **Perspective and scale match** — lens, distance, height | Wrong focal length reads as a wrong-sized person. | 🟡 ASSUMED for the reporter plate, must be recorded on set |
| 17 | **Colour pipeline** — Rec.2020 / ST2084, 1000-nit master, ProRes 4444 / DNxHR 444 for Resolve | | 🟢 validated end to end, including that the ProRes encoder silently drops `-color_trc` and needs the `prores_metadata` bitstream filter |
| 18 | **V-Log / V-Gamut decode on ingest** | His camera records log over a wide gamut. Undecoded, every downstream stage is wrong in a way that still looks like a picture. | 🔴 not built; constants not yet verified against Panasonic documentation |

Legend: 🟢 done and measured · 🟡 partial, with a known ceiling · 🔴 not built.

**Rows 1, 2, 3, 4 and 5 all require the same thing** — real geometry in a real renderer. That is
why the Cycles proxy stage is the architecture and not an enhancement.

---

## Orientation

```text
rounds/                               session handoffs — READ THE LATEST FIRST
music_videos/<video>/shots/<shot>/     shot.json + source key relight intermediate composite final logs
scene_library/<environment>/           .blend, probes/, background/, lighting_sh.json
footage/                               authoritative originals — never modify
assets/hdri/                           CC0 panoramas + provenance json
templates/  scripts/  logs/  dependencies/
```

**Every pipeline stage takes `--shot`.** There is no default; running a script without it lists
the available shots. A shot names its environment via `environment.dir` in its `shot.json`.

Three Python environments, deliberately separate:

| purpose | interpreter |
|---|---|
| pipeline (torch 2.8+cu128, opencv 4.13, numpy 2.4) | `dependencies/CorridorKey/.venv/Scripts/python.exe` |
| face landmarks (mediapipe 1.x needs opencv-contrib 5.x) | `dependencies/facetrack/.venv/Scripts/python.exe` |
| experimental EllSeg/RITnet inference (torch 2.8+cu128) | `dependencies/ellseg-runtime/.venv/Scripts/python.exe` |

Isolate any dependency that would disturb the pipeline's opencv — every EXR read depends on it.
Use `uv pip install --python <venv>/Scripts/python.exe <pkg>`; there is no `pip` in these venvs.

Blender is **5.2.1 LTS** at `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`.
A Blender Lab MCP add-on bridge may be listening on `127.0.0.1:9876`; `scripts/blender_mcp_client.py`
speaks it for read-only queries. Scene *building* runs under `blender -b --python`.

Hardware: RTX 5080, **16 GB, Blackwell sm_120** — requires CUDA ≥ 12.8 / torch ≥ 2.7. Any repo
pinning older torch will not produce a working kernel; it fails at runtime, not import.

---

## Blender 5.2.1 traps, verified in this repo

Each of these produces a wrong result with no error message.

1. **`scene.use_nodes = True` does NOT create the compositor node tree.** `scene.node_tree` is
   gone; `scene.compositing_node_group` stays `None` until you create one yourself:
   `bpy.data.node_groups.new(name, 'CompositorNodeTree')` and assign it.
2. **`CompositorNodeMixRGB`, `CompositorNodeMix` and `CompositorNodeMath` do not exist.** The
   compositor node set was restructured and there are no Mix or Math compositor nodes at all.
3. **Render Layers pass sockets are named in full words** -- `Diffuse Direct`, not `DiffDir`.
4. **Cycles has NO subsurface passes, and round 1's note to the contrary is wrong.**
   `use_pass_subsurface_direct/indirect/color` exist on the ViewLayer and accept `True`, but the
   Render Layers node exposes only Diffuse and Glossy. **Subsurface is folded into the diffuse
   passes**, so a diffuse-based ratio carries subsurface response rather than dropping it. The
   inherited warning said the opposite and would have sent a round chasing a pass that is not
   there.
5. **`OPEN_EXR_MULTILAYER` is rejected unless `image_settings.media_type` is
   `'MULTI_LAYER_IMAGE'` first**, and the error names the very tuple that contains the value.
6. **opencv cannot read multilayer EXR** -- `cv2.imread` returns `None` with no error, so any
   preview of a multilayer render must be written from inside Blender.
7. **Mesh emitters ARE visible to camera rays** -- the exact opposite of light objects, which are
   invisible even with `visible_camera = True`. Both halves of this bit: a light object threw
   shadows while staying invisible, and later an emissive bulb rendered a 266x187 disc of solid
   emission straight into an irradiance pass, which the composite then multiplied into the
   plate. **Anything an irradiance pass can SEE ends up scaling the photograph.**
8. **`use_persistent_data = True` serves stale results**, and the error GROWS through a sequence:
   rendering four frames with geometry and lights changing, persistent vs not, diverged by max
   7.55, 12.49, 14.83, 14.81 with mean error reaching 1.335. Confirmed, not inherited.
9. **`render.fps` is an int16**, so 60000/1001 silently clamps. Use `fps = 60, fps_base = 1.001`.
10. **`cv2.imwrite` to EXR rejects a float16 ARRAY.** Pass **float32** and let
    `IMWRITE_EXR_TYPE_HALF` do the conversion. Handing it a `np.float16` array warns
    *"Unsupported depth image for selected encoder is fallbacked to CV_8U"* and then asserts on
    `CV_8U` -- which reads like a codec problem rather than a dtype one. It cost a full 150-frame
    run, and the run reported **exit code 0** because the traceback was downstream of a pipe:
    check the tail of a background log, not just its status.
11. **numpy 2.x removed `ndarray.ptp()`.** `np.ptp(a)` is fine; `a.ptp()` raises `AttributeError`.
    The pipeline venv is numpy 2.4.
12. **`scene.render.filter_size` does not set Cycles' pixel filter width.** In 5.2.1,
    setting it to .01 leaves `scene.cycles.filter_width` at 1.5. Set and verify the
    Cycles property explicitly when a diagnostic specifies its pixel footprint.
    Round 21 caught this with textured eyes; round 20's constant-emitter checks
    away from boundaries did not expose it. See `rounds/round21.md`.
13. **Cycles BOX filtering forces width 1.0 internally**, even when
    `scene.cycles.filter_width` reads back .01. The installed UI hides the width
    for BOX, and Cycles' Blender sync overrides it. Round 22 caught this with
    independent textured transport before final-test scoring. For a narrow
    diagnostic filter use BLACKMAN_HARRIS, set its actual Cycles width, and model
    its sampling footprint. A property assertion alone does not verify rendering.
    See `docs/r22-eye-protocol.md` and `rounds/round22.md`.

## GPU: what is accelerated, and why the card still looks idle

Measured on the RTX 5080, 2026-09-08. The artist noticed the GPU sitting at 11% and 62 W of a
400 W limit during a Cycles render and asked whether anything was actually accelerated.

**Already on the GPU:** Cycles through OPTIX, and every neural stage through torch (Sapiens2
normals in fp16, Video Depth Anything, CorridorKey). **Not on the GPU:** all the numpy/OpenCV
per-pixel maths, and video encoding. Note `cv2.cuda` reports **0 devices** in the installed
wheel, so no OpenCV call can be accelerated -- `scripts/gpu_image.py` uses torch instead, with a
CPU fallback that reports which device it actually used.

**But the idle card is not an acceleration problem, and this is the part that matters.** Cycles
render time barely depends on sample count:

| samples | persistent off | persistent on |
|---|---|---|
| 4 | 2.21 s | 1.89 s |
| 128 | 4.06 s | 2.90 s |

**About 2 seconds of every frame is fixed Blender overhead** -- scene sync, device upload, film
init, EXR write -- and only a fraction is ray tracing. A GPU that finishes its work in 0.3 s and
then waits 2 s reports low utilisation honestly. Confirmed from the other side: GPU 3.29 s vs CPU
5.01 s on the real proxy scene is only 1.5x, because both pay the same fixed cost.

Consequences, which are the opposite of the intuition:

* **Samples are nearly free. Spend them.** 128 samples costs 1.3 s more per frame than 4.
  Rendering noisy output to "save time" saves almost nothing.
* **Moving more work onto the GPU will not speed up rendering.** The GPU is not the constraint.
* **Fewer, larger renders beat many small ones.** Per-frame overhead is paid per render call.
* **`use_persistent_data = True` is NOT the way out.** It would save ~20%, and round 1 recorded
  that it serves stale lighting. That is now measured rather than inherited: rendering the same
  4-frame sequence with geometry and lights changing, persistent vs not, diverges and the error
  GROWS through the sequence -- max abs difference 7.55, 12.49, 14.83, 14.81 and mean error
  0.062, 0.186, 0.157, **1.335**. Do not enable it.

## House rules

- **GPU first.** The artist, round 38: *"Never run stuff on the CPU when it can be avoided."*
  New per-pixel or per-frame maths goes in torch on CUDA (`scripts/gpu_image.py` is the
  pattern; `cv2.cuda` has 0 devices in this wheel). A stage that is CPU-bound gets ported or
  parallelised when it is touched, and never grows a new single-threaded per-frame loop
  silently. He plans ~10 takes × 10 angles × 3-minute songs; at CPU speed that is days.
  Round 39, when I proposed vectorising the CPU eye pass instead: *"Port it to the GPU
  instead of optimizing it on the CPU."* A port is gated on reproducing the CPU run it
  replaces (`validate_eye_gpu.py`); `eye_gpu.py` holds cv2/numpy arithmetic reproduced bit
  for bit (round39.md §3). Tiny per-eye solves (IRLS on ~60 points) stay in numpy on the
  GPU's exact samples — rank-deficient steps are not reproducible any other way.
- **It is a generic tool.** *"It can accept any kind of input video."* Nothing may depend on
  this clip's earlier runs (an accepted canonical, a hand-set frame range) or on a constant
  that only held at one clip length — express it relative to the input (a fraction of frames,
  a window in SECONDS, not frames).
- **Verify conventions, never assume them.** Axis, handedness and sign errors in this domain are
  invisible in a single frame and ungradeable afterwards. Test several directions, not one — a
  reflection error leaves the diagonal invariant and passes a single-direction check.
  `scripts/validate_sh_convention.py` is the pattern.
- **Measure before and after.** Every claim in `rounds/` is backed by a number from the repo.
  Prefer a control (relight off) over an unanchored figure.
- **No on-set reference is ever a requirement.** No chrome ball, no grey card, no colour chart
  as a precondition. The artist will not do it, and this is meant to run on other people's
  footage where no reference exists. If something needs measuring, infer it from the plate:
  `estimate_source_light.py` is the pattern -- it replaced the pipeline's last hand-written
  constant by fitting SH lighting to skin radiance against the neural normals.
- **State assumptions as assumptions.** The reporter camera match and `E_source` are both
  assumed and labelled so in `shot.json`. Do not let an assumption harden into a fact.
- **Never recalibrate the proof-of-concept shot's keying thresholds.** They are hand-set because
  that plate has a green costume, and they produced the keys behind a validated master.
- Commit with a message that explains *why*, including bugs found and what they would have cost.
- The README carries pipeline detail and the full list of Blender/library traps.

---

## The pipeline, stage by stage

Every stage takes `--shot`. Interpreter is the pipeline venv unless the row says Blender.
`P` = `dependencies/CorridorKey/.venv/Scripts/python.exe`.

| # | stage | script | writes | notes |
|---|---|---|---|---|
| 1 | ingest / extract frames | `shot_paths.py` resolves everything | `source/frames/` | rates are exact `Fraction`s |
| 2 | calibrate key | `calibrate_key.py` | `logs/key_calibration.json` | thresholds do **not** transfer between plates |
| 3 | key + despill | `key_plate.py` | `key/alpha/`, `key/fg/` | despill is the manual pass; CorridorKey's own `despill_strength` never touches `result['fg']` |
| 3b | ~~specular / diffuse split~~ | `separate_specular.py` | `key/specular/*.exr` | **EXPERIMENTAL, REJECTED — not in the delivery path.** Kept for diagnosis only; see below |
| 4 | depth | `video_depth.py` | `relight/depth/*.npy` | Video Depth Anything; **affine-invariant disparity**, not metric |
| 5 | face track | `face_track.py` (facetrack venv) | `relight/face_track.json`, `relight/face_landmarks.npz` + `.json` | mediapipe; needs its own venv. Also stores ALL 478 landmarks, blink, eyelid aperture, head pose |
| 5b | **corneal geometry** | `eye_geometry.py` | `relight/eye_geometry.json` | corneal sphere per eye, and the PLATE's key light solved from its own catchlights |
| 5c | **limbus track** | `limbus_fit.py` | `relight/limbus_track.json` | the limbus fitted to the IMAGE EDGE, one radius per shot, no smoothing. `--device cuda` (default) = the CPU track exactly (round 39). Audited by `audit_r35_limbus.py`; drawn on every frame by `review_r35_track.py` |
| 5c′ | **limbus trust** | `limbus_trust.py` | `relight/limbus_track_tv.json` | bridges a run of high-SE frames from trusted frames either side (offset from the eye corners) ONLY where the image's edge support prefers the bridge, per run (round 39 §2a). Then `eyelid_fit.py --update` refits just those eyes' lids |
| 5d | **eyelid track** | `eyelid_fit.py` | `relight/eyelid_track.json` | both lid margins fitted to the IMAGE, robust parabola per lid, curvature one per eye per shot — or pooled over `--curvature-window` SECONDS with anatomical sign only, which a long clip needs (round 38 §0). `--device cuda` (default) = the CPU track exactly; `--update <track>` refits only eyes whose limbus centre moved. `--lower-iris-edge` (round 40) adds `lower_iris`, the lower lid where it covers her iris, which the walk cannot see (`lower` unchanged). **Not the paint bound** — see `lash_ceiling()` |
| 6 | **normals** | `neural_normals.py` | `relight/normals/*.npz` | see below |
| 7 | **albedo + skin mask** | `make_albedo.py` | `relight/albedo/`, `relight/skin_mask/` | plate ÷ `E_source(normals)`; needed before the plate can be re-lit, or it is lit twice |
| 7b | **proxy geometry** | `make_proxy_mesh.py` | `relight/proxy/*.npz` | see below; carries UVs |
| 8 | environment | `build_hdri_scene.py` (Blender) | `scene_library/<env>/` | anamorphic path is opt-in |
| 9 | light probes | `bake_light_probes.py` (Blender) | `<env>/lighting_sh.json` | SH-L2, Lambertian-convolved |
| 10 | relight | `relight_sh.py` | `relight/controls/*.npz` | gain map; **has a measured ceiling**, see round 2 |
| 11 | proxy render | `render_proxy.py` (Blender) | `relight/proxy_render/*.exr` | self-shadows, SSS, glossy passes |
| 11b | stress demo | `stress_proxy.py` (Blender) | `composite/stress_proxy*/` | lightbulb tour; `--textured` uses the albedo |
| 11c | **ratio irradiance** | `stress_proxy.py --ratio` (Blender) | `composite/ratio_render/*.exr` | neutral albedo at PLATE resolution and the PLATE lens |
| 11d | ~~catchlight~~ | `stress_proxy.py --catchlight` (Blender) | `composite/catchlight_render/*.exr` | renderer looks sound and its validator passes; **the compositing of it is rejected and `--catchlight` defaults to 0** |
| 11e | **corneal catchlight** | `render_eye_catchlight.py` (Blender) | `composite/r34_catchlight/*.exr` | the scene's own reflection in a corneal cap; validated by `validate_eye_catchlight.py`. Parked in round 34; UNPARKED in round 41 for the tour (below) |
| 11e′ | **corneal reflection of the tour** | `stress_proxy.py --corneal` (Blender) | `composite/r41_corneal/*.exr` | round 41: the same cap lit by the bulb tour, anchored on the fitted iris centre behind the measured gaze (`corneal_anchor.anchor_limbus`). Checked by `validate_tour_catchlight.py` (`--plate-glint` grades the sphere against the plate's own softbox glint; `--render` against an independent reflection model with wrong-convention controls) and `measure_glint_psf.py` (plate vs render sharpness → `--corneal-blur`) |
| 11e″ | **wet eye: globe + socket** | `stress_proxy.py --corneal --globe --socket` (Blender) | `composite/r42_eye_pass/*.exr` | round 42: ONE surface of revolution per eye -- the corneal sphere, a 1 mm concave limbal fillet, and the 12 mm scleral globe (tear film over conjunctiva, rougher lobe, `--sclera-roughness` ASSUMED 0.08: the plate's softbox reflects in the globe UNDER the cornea on every open eye, so nothing fits it) -- inside a black head-attached SOCKET cup (mediapipe head pose, visual-field limits nasal 60 / temporal 100 / up 60 / down 75 deg, 24 mm) so a source behind or beside her head stops reflecting. Validated by `validate_tour_catchlight.py --render <pass> --globe` (bulb reflected in the globe vs the nearest rendered peak outside the junction, wrong conventions alongside). Composited with `--corneal-matte lids` |
| 11g | **scleral glint removal** | `sclera_glint_removal.py` (GPU) | `composite/<name>/` (eye-pass layout: removal/ weight/ source/) | round 42: the capture room's reflections on the WHITE of her eye (the blob `audit_r42_wet.py` finds outside the limbus, inside the fitted opening), footprint held in time, filled from the sclera around it, applied ON TOP of the accepted iris run and its weights unioned; corner (tear-lake) glints left unless `--corners`. `composite_ratio.py --eye-removal <name>` then substitutes both |
| 11f | **eye: one canonical iris** | `eye_removal_gpu.py` (GPU, same CLI; `eye_soft_removal.py --fill-iris` is the CPU reference) | `composite/<name>/` | **Round 40's command is in `rounds/round40.md` §7** (round 39's plus `--lid-floor` on `eyelid_track_wtv40.json`; `removal.json` names each dropped eye's `drop_stage`). **Round 39's command is in `rounds/round39.md` §7** (adds `--lid-claim`, `--skin-gate`, the trusted tracks, `--dump` for `audit_r39_classes.py`); the GPU pass refuses flags it does not implement. Built once in the registered eye frame. **Round 37's accepted command is in `rounds/round37.md` §7** (softbox region, parabola lash line, rebuilt canonical, temporally filtered edge shading, limbus gate, lid bridging rules). `--lid-shadow` is DEFAULT OFF (round 36 §4). Holds every frame in RAM packed by `frame_store.py` (≈12 MB/frame, bit exact); the full-clip command is `rounds/round38.md` §7. Audited by `audit_r37_eyes.py` + `audit_r37_flicker.py`; **looked at** with `review_r37_sheets.py`, `review_r37_temporal.py`, `review_r37_panel.py` |
| 12 | **ratio composite** | `composite_ratio.py` | `composite/ratio/` | `out = plate × R` by default. Round 41: `--eye-removal <run>` substitutes the eye pass's pixels under the ratio (exact fg outside its weight); `--corneal-dir <pass> --corneal-blur 0.35` ADDS the corneal reflection through the fitted lids and limbus (mediapipe polygon where no lid fit, nothing where no limbus fit) in the ratio's own anchor units. Round 42: `--corneal-matte lids` drops the limbus gate for a `--globe` pass, whose wet sclera reflects too (the matte is then the fitted lids only). `--plate-specular remove` and the round-5 `--catchlight` are rejected experiments |
| 12b | composite (gain map) | `composite_shot.py` | `composite/`, `intermediate/linear_4k/` | legacy path; loads background **per frame** |
| 12c | **review video** | `review_video.py` | `final/*.mp4` | what the artist watches; `--compare` puts the before on the LEFT with both halves labelled |
| 13 | master | `master_export.py` | `master/*.mov` | ProRes 4444 / DNxHR 444, 4:4:4 10-bit |
| 14 | delivery | `hdr_mux.py` → `validate_final.py` | `final/*.mp4` | HEVC 4:2:0 PQ + HDR10 SEI |

### Showing work to the artist (stage 12c)

**A round is not shown until there is a video.** Frames on disk are not a result he can judge —
the artefacts that matter here are temporal, and his eye is the acceptance bar. Rounds 1–4 each
re-derived the same ffmpeg line and re-probed the previous round's file for its settings; that is
what `review_video.py` exists to stop.

* Review encodes run at **24 fps regardless of the shot's rate**. The stress tour is 150 rendered
  frames, not 150 frames of a 59.94 fps performance. Pass `--fps shot` when judging real motion.
* They are **review only** — display-referred sRGB, `crf 12` because a normal crf hides the
  sub-pixel detail a round is usually arguing about. Delivery stays `master_export` → `hdr_mux`;
  never hand a review file to a grade.
* `--compare <dir>` puts the before on the LEFT and burns a label into each half. Do that by
  default when a round claims an improvement: a claim of "the dots are gone" is worth far less
  than two halves moving side by side.

### How normals are computed (stage 6)

`facebook/sapiens2-normal-1b` via `transformers` in the **pipeline venv** (needs ≥ 5.16; pin
numpy at 2.4.2 when upgrading, or it drags in 2.5.3 and disturbs opencv). ~0.42 s/frame on the
5080.

Three input requirements, none optional:
1. **Crop to the matte bbox, expanded to 3:4.** A raw 16:9 frame letterboxes into the model's
   1024×768 and leaves the face at ~150 px; the crop puts it near 600 px.
2. **Composite onto neutral mid-grey.** A saturated green backing is out of distribution and
   bleeds into the estimate exactly where hair detail lives.
3. **Convert linear → sRGB** before ImageNet normalisation. The keyed foreground is scene-linear.

**The camera→world convention is `world = (c₀, −c₂, c₁)`** — an axis swap, measured, recorded in
`logs/normals_convention.json`. `neural_normals.py` refuses to run without that file. Re-run
`validate_normals_convention.py` if the model, the checkpoint or the scene camera ever changes;
it ranks all 48 signed permutations against a sphere and corroborates with flip equivariance and
anatomy.

**What normals are for, and what they are not.** They give surface orientation, so they drive
`dot(n, l)` — *shading*. They cannot produce *shadows*: a cheek pixel's normal contains nothing
about the nose in front of it. Round 2 measured that swapping depth-gradient normals for neural
ones did **not** improve directional lighting. Their real value here is coherence, correct
extinction when lit from behind, and supplying the fine relief the proxy mesh needs.

### How proxy geometry is built (stage 7)

A displaced silhouette shell, in Blender world axes (+X right, +Y away from camera, +Z up, actor
at Y = 0):

* silhouette from the matte, displaced by metric depth using the **same** back-projection and
  calibration as the normals, so the two cannot disagree about where the actor is;
* closed with a flat back cap and stitched side walls, or it is invisible to rays from behind;
* **relief scale anchored on the face** at 0.085 m of visible-surface depth. Raw monocular depth
  over-estimates facial relief by ~2.2× and produced a 68 cm-deep bust;
* **fine relief integrated from the normals** (Poisson, DCT, Neumann). For `Y = f(X,Z)` the
  normal *is* the gradient field. Depth supplies scales above 5 cm, normals below — their error
  profiles are opposite. Without this the render is a mannequin with no nose ridge.

Validate with `validate_proxy_shadows.py`: it renders an uplight and a toplight and requires the
shadow to *move across the nose*, which is the one thing no per-pixel method can fake.

### The ratio composite, and why it is the delivery path (stage 12)

    out = plate × R,      R = E_target(rendered) / E_source(assumed)

**Never render the performer and show that.** A proxy has no pores, no individual hair strands,
no fabric weave, no corneal bulge. Rendering it produces exactly what the artist named on sight:
"a Vision Pro avatar", "a Sims character", "a plastic Barbie doll", "a 90s CGI game trailer". His
frame-by-frame verdicts mapped precisely onto how much of her the light revealed — heavily
shadowed frames read as acceptable, broadly lit ones did not — because a broad light exposes the
detail a smooth shell does not have.

The ratio composite renders the *irradiance field* on a neutral proxy, divides by the plate's own
assumed lighting, and multiplies the photograph. Every pore, hair and specular in the output is a
photographed pixel; the renderer contributes only the CHANGE in light, which is smooth almost
everywhere and sharp exactly where it should be — at a shadow edge.

Two rules that are easy to get wrong:

* **The ratio pass must use the plate's own lens and resolution**, or the multiplier does not
  line up with the photograph it multiplies. The wider demo lens in `stress_proxy.py` exists only
  to keep the bulb in frame and must not leak into `--ratio`.
* **Clamp the top of the ratio, not the bottom.** A shadow going dark is the point. `relight_sh.py`
  clamped both ends at 1.5 stops, capping the whole map at 2.83:1, which is exactly why it could
  never produce a terminator.

What it still inherits from the proxy: the *shape* of the shadows, so a nose shadow lands in
roughly the right place with the proxy's smoothness at its edge, and hair shadows as a mass. What
it does not inherit is the proxy's appearance, which is the whole difference.

### The specular layer (stage 3b) — BUILT, REJECTED, AND OFF. Read this before trying again.

`plate = D + S`. The ratio is a lighting scale and **only D obeys it** — a smooth `R` can recolour
a specular highlight but cannot move, reshape or delete one. So before round 5 every highlight in
the plate kept reporting where the green-screen room's lamps were, no matter where the Blender key
went. The artist reported this three times, in three places, without connecting them:

* *"I can kind of see the shine in her hair from the original green screen room."*
* *"the lips are somehow lit up more than the skin around them"*
* the corneal catchlight, still sitting at 11 o'clock on both irises — a plainly rectangular
  softbox, in a composite lit by a bulb.

**They are one bug** — and round 5's attempt to fix it made the picture much worse. The artist's
verdict on the result was *"It looks extremely terrible - all of it"*, and the measurements agree:
**26.9 % of tooth energy and 52.7 % of sclera energy removed**, 75.9 % of sclera pixels losing
over a quarter of their radiance, dark hair breaking into unstable patches. `--plate-specular`
defaults to `keep`. The stage is preserved for diagnosis and must not be re-enabled because the
files exist.

**The diagnosis that looked obvious was wrong, and that is the useful part.** It is tempting to
blame the neutral guard (`--cos-max`, a 1.8° cone) for failing to protect near-neutral teeth and
sclera. Measured, their local base colours sit **41.4° and 28.6°** from the illuminant — nowhere
near the guard, and **0 %** of sampled tooth or sclera pixels were inside it. Widening the cone
fixes nothing. The real fault is that **a local morphological opening is not a valid
material-colour reference**: it mixes materials across its window, so `b̂` is wrong wherever a
window straddles a boundary — which is exactly the mouth, the eye and every hair edge. Likewise
the hair blocks are not literal kernel stamps (measured patches are 5×4–7×4 px on a 15 px
kernel), though removal does swing 11.5 / 16.7 / 9.4 % across kernel widths 7 / 15 / 31.

If you try again, the bar is: **prove teeth, sclera and hair detail survive, and that the result
is temporally coherent, BEFORE enabling it.** `scripts/audit_r5_defects.py` is that test.

The parts below are what the attempt got right and are worth keeping:

* **The illuminant is measured, not assumed.** From the performer's own corneal glints — the eye
  is the plate's own chrome ball — cross-checked against the brightest near-neutral pixels. On
  the reporter plate the two agree to 0.3% while sharing no pixels and no mechanism.
* **The material colour is LOCAL**, from an alpha-aware opening, so there is no skin mask, no hair
  mask, and no assumption about skin tone, hair colour or wardrobe. Each pixel is fitted as
  `I = a·b̂ + s·Ĝ` with `a, s ≥ 0`: a rise in red alone (blood, lipstick) can only be `a`.
* **Skin gets *more saturated as it darkens*** — subsurface reddening — which mimics the
  dichromatic specular signature exactly. Measured over 58 819 cheek pixels of one material,
  `min/sum` runs 0.060 (p2) to 0.216 (p95). That is a genuine obstacle for any dichromatic
  method here: **a single unpolarised image cannot fully separate diffuse from specular on
  skin.** Cross-polarised capture can, and the artist will not shoot cross-polarised.

Near-neutral materials — a white shirt, grey foam, metal — are unseparable by *any* dichromatic
method. Round 5 reported that honestly for the wardrobe and then destroyed the teeth and sclera
anyway, because the failure there was not neutrality at all.

### Catchlights (stages 5b, 11d) — renderer sound, compositing REJECTED, `--catchlight` = 0

**The corneal geometry and the Blender render are validated** (`validate_catchlight.py`, direction and magnitude within 0.44–3.51 %). **The compositing of it is not.** The cap is a flat 2D disc centred on the iris with **no eyelid intersection**, so 5.8 % of its weight lands outside the eye opening and it painted silver arcs across her upper lids. Fixing the lid mask alone is not enough: source occlusion is still wrong (the proxy has no eye sockets) and it depends on the rejected removal stage to take the old reflection out.

**Do not paint a dot.** A catchlight is a mirror image of the source, so the honest way to make one
is to put a mirror where the cornea is and let Cycles find the source. Presence, size, shape,
colour and occlusion then all come from the scene: a bulb 0.5 m away gives a bright disc, a window
a soft rectangle, an overcast sky a broad wash, an unlit corner nothing at all.

**A distance threshold would get sunlight exactly backwards.** A mirror shows the source's
*radiance*, and radiance does not fall off with distance — so a distant lamp still makes a
catchlight, just a small one, and the sun makes a tiny brilliant spark rather than nothing. What
distance changes is solid angle, hence size.

**The cornea is a dielectric, not a mirror.** IOR 1.376, so Fresnel gives ~2.5 % at normal
incidence. A first attempt used a plain glossy BSDF with a white colour, i.e. 100 % reflectance,
and returned the whole cornea at 2.5–4.3 linear — it would have whited out both eyes.

**Keep or replace is computed, not set.** The plate's own catchlight is better than any synthetic
one when it fits — right shape, right softness, real lash occlusion. It fits only when the new key
is near the original. Both angles are measured: the plate's from its own catchlights
(`eye_geometry.py`, 54.6° off the camera axis on the reporter plate) and the scene's from the L1
moment of the rendered irradiance. Note the glint moves by **half** the angle the light moves, so
the tolerance looks generous and is not.

**Gaze estimation is resolution-dependent, and this is measured.** The iris ellipse's minor/major
ratio reads gaze directly, but needs a big iris: at 12.9 px radius it returned a confident, stable
**29.3° for a subject looking straight down the lens**, because one pixel of landmark asymmetry on
a 13 px circle is already 23°. It poisoned the light solve (40° scatter) before it was caught. So
the ellipse is used only above `--ellipse-min-radius` (the artist's macro close-up case) and head
pose plus iris displacement is the default below it. On the reporter plate that took the scatter
from 40° to 18.5°.

**The light POSITION does not come out**, and the gate for that is not the residual. Two corneas
63 mm apart cannot triangulate a source a metre away: the rays are near parallel and the
least-squares point collapses onto the corneas themselves *with a small residual*. The direction
is the usable output.

### Anamorphic (`scripts/anamorphic.py`)

All 1.6× arithmetic lives in one module; never retype the factor. Opt in per shot with
`camera.anamorphic` in `shot.json` — the stock reporter plate is spherical and must not inherit
it. Geometry is exact (3840×1600 desqueezed = the full 36×24 mm gate squeezed); bokeh direction
was measured, not assumed (`aperture_ratio` 1.6 → 1.6:1 vertical oval).
