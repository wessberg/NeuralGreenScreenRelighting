# Round 40 handoff — 933–955 right was the lower lid, not gaze; the real off-axis class is the glance

**Date:** 2026-09-10. Starts from round 39 (`r39_fixF`, *"looks good"*). Production defaults
unchanged — every new behaviour is a flag; the command is in §7.
**Shown to the artist:** `reporter_full/final/r40_floorB_915-975_6fps.mp4` (SOURCE / R39 / R40,
the changed stretch at 6 fps) and `final/r40_floorB_eyes_phone.mp4` (the full 1:40). **No
verdict recorded yet.** His one instruction this round, when I had moved on to an elliptical
limbus for the glance frames: *"I think maybe that should be part of a fresh round in a fresh
session"* — so that is round 41 (§6, §8), with this round's probe data ready for it.

---

## 0. What was built

* **`eyelid_fit.py --lower-iris-edge`** — a SEPARATE `lower_iris` fit per eye: where her
  lower lid covers the iris. Per column of the lower walk (same columns and samples), the
  first up-crossing of halfway between HER iris (median of the 0.45–0.85 annulus, specular
  out) and HER lower-lid skin (75th percentile of the 1.3–1.65 band below the centre), while
  both samples are ≥ 0.35 r out and ≥ 1.5 px inside the limbus, the specular blocked, held
  above for 1 px, and a real step (gradient > MIN_STEP). No rule where her iris and skin do not
  separate by 0.12 display (a light iris). Fitted at the eye's lower-lid curvature, stored with
  `u_span` (its inlier columns). **`lower` itself is untouched**: the new track minus
  `lower_iris` equals `eyelid_track_w.json` on all 2400 eyes; GPU = CPU exactly
  (`track_gpu.iris_edge_samples`, `eyelid_fit.iris_edge_samples`). `--update` inherits it
  from the base track.
* **`eyelid_fit.py --lower-spec`** — block the specular for the lower walk as the upper one
  always has. Built and exact; **not used** (§4).
* **`eye_removal_gpu.py --lid-floor`** (`--floor-margin`, default 0) — the mirror of the lash
  line: the fill is bounded from below by the `lower_iris` fits. A **per-column median over the
  frame and its two neighbours, a missing floor counting as no bound** (an isolated fit is
  never a floor; a one-frame gap takes the lower of its neighbours; nothing is interpolated
  across a run without one), each fit held to its own `u_span` and released 1 px/px past it.
  Recorded per eye as `floor_pixels`. It changes only `final_weights` — tracks, canonical,
  sigma, gain and every gate upstream are identical.
* **`drop_stage`** in `removal.json` for every `no_eye_frame` eye: `lid_opening_under_90px`,
  `inside_under_60px`, `anomaly_not_found`, `no_limbus_or_window`, `no_canonical`.
* `review_r39_classes.py` draws the floor (white); `--dump` stores it per column.
* **`scripts/probe_r40_ellipse.py`** — DIAGNOSTIC for round 41 (§3).

## 1. Results (`r40_floorB` vs `r39_fixF`)

Regression gate first: the pass on the new track WITHOUT `--lid-floor` reproduces `r39_fixF`
exactly (1200/1200 frames, 0 decision differences, canonical identical;
`logs/diff_r40_chk.json`).

With the floor: **758 frames identical, canonical identical, exact source outside the
weight** (`logs/diff_r40_floorC_vs_B.json` also shows the drop_stage refactor is identical).
**15 right eyes go from softbox-only to the full fill** — 584, 836, 933–935, 947–955, 977 —
and **nothing else changes status anywhere**. Material pixel changes on 24 eyes, all looked at
(`composite/r39_review/r40B_*`), and 928–961 R stepped frame by frame: R39 switched her right
eye from the painted iris to her own and back (933–935, 947–955); R40 holds the painted iris.

| ungated, from HER image (`lower_bound_audit`, §5) | r40_nogate (R39 lids) | **r40_floorB_ng** |
|---|---:|---:|
| right: fill over skin below the centre, eyes ≥ 6 px / total px | 18 / 333 | **7 / 152** |
| right: her iris left unfilled under the fill, total px | 26020 | 26106 (+86) |
| left: fill over skin, eyes ≥ 6 px / total px | 15 / 221 | 13 / 213 |

| `audit_r37_eyes.py`, 2107 eyes | r39_fixF | **r40_floorB** |
|---|---:|---:|
| treated | 2016 | 2016 |
| flicker p95 / max | 0.0354 / 0.328 | 0.0350 / 0.328 |
| departure p95 / max | 0.0357 / 0.152 | 0.0363 / 0.152 |
| lash eyes / px | 207 / 452 | 209 / 461 |
| region standout > 0.05 | 77 | 77 |

`audit_r39_classes.py`: painted skin (≥ 6 px) 0 / 0; skin px total 120 → 126 (the 15
newly filled eyes, each under the gate); softbox classes unchanged. Skin gate now takes 12 L /
8 R eyes (R39: 12 / 22). `logs/r40_eyes_audit_B.json`, `logs/r39_classes_r40_floorB.json`.

**The left eye is barely helped** (221 → 213 px): its remaining skin cases (226–230, 245,
865–867, 1111 L) found too few iris-edge columns. Look at them before tuning anything.

## 2. Round 39's e4 diagnosis at 933–955 right was wrong — measured

Round 39 filed 933–955 R as "off-axis gaze: the canonical is a circle, her foreshortened iris
is not". Over all 2400 eyes of the UNGATED pass, **76 % of every pixel of skin the fill painted
sits below the limbus centre at rho 0.62–0.72** — deep inside any plausible ellipse. An ellipse
fitted to the limbus track's own edge points finds a 7–10 % VERTICAL squash on nearly every
frame (b/r median 0.93 L / 0.90 R, minor axis ~80–100°; probably the lids biasing the arc)
and nothing horizontal at 933–955 R. It was **the lower lid**: `eyelid_fit`'s lower walk takes
only bright-to-dark steps (sclera into the lid margin); iris into lid skin is dark-to-bright,
so under the iris every column returned NaN (947 R: 956–960.5 found, 961–990.5 nothing) and
the lid there was an extrapolation of the sclera columns, 3–5 px low.

## 3. The real off-axis class: extreme sideways glances (round 41's work)

Found by `drop_stage`: the open eyes the pass drops at `lid_opening_under_90px` are **225,
961–964, 1050–1054 L** — the glances. Drawn (`pass_lids_view`, §5): **the limbus circle is not
on her iris.** At a hard glance the iris is foreshortened — its dark disc ~14–16 px wide
against the 24.5 px shot diameter — and the fixed-radius circle sits ~5 px to one side with
half of it on sclera; the lids fitted around it cut the opening. `--lower-spec` fixes the lid
but not the circle (the eyes still drop). The head turn (110–214) is the extreme of the same
thing: the iris a narrow sliver in the eye corner, the circle fit abstaining
(`too_few_edges`, `limbus_occluded`) or, at 125 L, sitting on the sclera.

**`probe_r40_ellipse.py`** (axis-aligned ellipse, vertical semi-axis r, horizontal b in
0.45r–r, Daugman along the contour normal, lid-free angles, vs the best circle in the same
±8 × ±3 px window):

| eyes | best ellipse b/r | gain over best circle |
|---|---:|---:|
| glances 961–964, 1050–1054 L | ~0.63, centre ~4 px toward the corner | **+23 … +37 %** |
| 125 L (extreme corner) | 0.45 (search floor) | +197 % |
| ordinary eyes (700 sampled, every 3rd frame) | 1.00 | **p50 0.0 %, p90 0.0 %, p95 3.1 %** |
| **beside a glance** (954–960, 965, 1049 L) | 0.63 displaced the OTHER way | +3 … +8 % — AMBIGUOUS |
| also ≥ 15 %: 30–36 R, 123 L, 537 R, 1164 R | 0.52–0.8 | not looked at yet |

The pass already paints an axis-aligned ellipse when the track supplies ax ≠ ay (rho, theta,
`predict`, `sample_eye` all use ax/ay) — **what is missing is the fit**, and a way to reject
the ambiguous narrow ellipses (likely the pupil edge on one side, the limbus on the other).
Data: `logs/r40_ellipse_probe_glances.json`, `logs/r40_ellipse_probe_every3.json`.

## 4. Tried this round and set aside

| try | why |
|---|---|
| paint the canonical as an ellipse at 933–955 R (round 39's plan) | the skin there is at rho 0.62–0.72 below the centre; no ellipse reaches it (§2) |
| fold the iris edge INTO the lower-lid parabola (`r40_lidA`) | moved ~400 eyes; the lid inset reads **max(upper, lower) rms**, so the new lower fits widened the inset on 79 eyes top and bottom; a tilted parabola through a few iris columns cut into open irises; and **1050 L came un-dropped onto a glance-displaced circle — a second pupil, round 39 §2a again** |
| `load_lids` bridging for the floor | carried two isolated single-frame fits (858 R on a blink, 866 R) across 862 R and cut an open iris |
| the floor's parabola across all columns | a tilted fit's extrapolation climbed into the disc (21–22, 386, 836–837 L); hence `u_span` |
| `--lower-spec` in the track | right at 717, 394, 1051 R (the softbox's lower edge had been taken for the lower lid, a diagonal through the iris) and 401/405 L (an inner-corner speck), but steep wrong lids on blink frames (852, 128 L, 391 R) and it un-drops 1050 L. Needs the ellipse limbus first |
| a gate on limbus rms / SE ratio for 1050 L | no clean gap (866 L at 19.8× rms is treated and fine; 23 L at 16× SE was accepted) |

## 5. Metrics and facts worth not re-deriving

* **The lid inset couples the lids**: `esr.lid_inset_for` = 2 + 2 × max(upper rms, lower rms)
  is applied to BOTH lids. Any change to one lid's fit quality moves the other's bound.
* 76 % of painted-skin pixels are below the centre (the lower lid); upper 11 %, sides 13 %.
* `lower_iris` fits: 315 L / 474 R eyes; 5 L / 6 R isolated (no neighbour with one).
* `drop_stage` of open eyes (mediapipe aperture ≥ 0.8) in `r40_floorC`: L —
  `lid_opening_under_90px` 225, 961–964, 1050–1054; `anomaly_not_found` 124, 125, 622, 645;
  `no_limbus_or_window` 14, 110, 119–121, 129–130, 136, 200–212, 540, 705, 828, 1101–1103.
  R — `no_limbus_or_window` 120–129, 197–214 (the far eye near profile), 369, 379, 452, 594,
  618, 705, 1100–1102, 1141–1149; `anomaly_not_found` 827.
* The artist's other project (`eye-clean`, same footage) was running on the GPU this round:
  a full eye pass took 751 s instead of ~90 s. Check `nvidia-smi` before reading a time.
* Scratch diagnostics used this round (not committed; worth scripting if reused):
  `lower_bound_audit` (fill over skin / iris left under the fill, per eye, from HER image),
  `eye_changes` (per-eye weight and pixel change between two runs), `pass_lids_view` (the
  pass's own bridged, filtered lids + inset + visible px, drawn), `lid_samples_view` (the
  column samples each lid walk found).

## 6. Still open — the round 41 list

1. **His verdict on `r40_floorB`** outranks everything below.
2. **The glance: an elliptical limbus** (§3). Fit b (and centre) where the image demands it,
   reject the narrow ellipses beside a glance (check the pupil sits at the ellipse's centre, or
   constrain by where the iris sits between the corners, or temporal continuity of b), write
   `smoothed_axes = [b, r]`, let `eyelid_fit --update` refit those lids (it uses `got[1]` =
   ax as its radius — use the vertical axis), and look at 961–964, 1050–1054 L, 125 L, 30–36 R,
   537 R, 1164 R. The iris-edge floor's inside-the-disc test is a circle — make it the ellipse.
   **Look at every eye it moves against r40_floorB before sending**; 1050 L is exactly where
   round 39 §2a painted a second pupil.
3. Then `--lower-spec` becomes safe to evaluate again (§4).
4. The left eye's remaining lower-lid skin cases (§1).
5. Round 39's list otherwise stands: surviving softbox on open eyes (225, 295 L; the far eye
   near profile), the class audit's specular detector missing motion-blurred highlights,
   the canonical's softness, scleral glint only if he asks, keying speed.

## 7. Reproduce

```powershell
$P = "dependencies/CorridorKey/.venv/Scripts/python.exe"
$F = "music_videos/reporter_test/shots/reporter_full"
# (frames, key, face track, limbus_fit, limbus_trust as round39.md section 7)
& $P -B scripts/eyelid_fit.py --shot $F --curvature-window 2.5 --lower-iris-edge --out eyelid_track_w40.json
& $P -B scripts/eyelid_fit.py --shot $F --track limbus_track_tv.json --update eyelid_track_w40.json --out eyelid_track_wtv40.json
$base = @('--shot', $F, '--count', '1200', '--canonical-votes-frac', '0.44', '--fill-residual-gate', '8',
  '--fill-iris', '0.99', '--grain', '0', '--pupil-target', '0.34', '--pupil-colour',
  '--limbus-gate', '1.0', '--lid-temporal', '--lid-line', 'parabola', '--lid-line-window', '2',
  '--specular-reach', '2.0', '--no-lash-exempt', '--region-rows', '--top-shading', '2.5',
  '--shading-sides', 'all', '--shading-temporal', '2', '--gain-ring', 'lower', '--gain-temporal',
  '--level-window', '2')
# the gate: without --lid-floor the new track reproduces r39_fixF exactly
& $P -B scripts/eye_removal_gpu.py @base --name r40_chk --dump --limbus-track limbus_track_tv.json `
     --eyelid-track eyelid_track_wtv40.json --lid-claim 0.8 --skin-gate 6 --skin-gate-low 3
& $P -B scripts/validate_eye_gpu.py --shot $F --ref r39_fixF --test r40_chk --out "$F/logs/diff_r40_chk.json"
# R40 (r40_floorB):
& $P -B scripts/eye_removal_gpu.py @base --name r40_floorB --dump --limbus-track limbus_track_tv.json `
     --eyelid-track eyelid_track_wtv40.json --lid-claim 0.8 --skin-gate 6 --skin-gate-low 3 --lid-floor
& $P -B scripts/audit_r39_classes.py --shot $F --name r40_floorB --limbus-track limbus_track_tv.json
& $P -B scripts/audit_r37_eyes.py --shot $F --names "r39_fixF,r40_floorB" --eyelid-track eyelid_track_wtv40.json --out r40_eyes_audit_B.json
& $P -B scripts/review_r39_classes.py --shot $F --name r40_floorB --ref r39_fixF --items "933:right,947:right" --neighbours 3 --cols 7 --out look
& $P -B scripts/review_r35_eyes.py --shot $F --dirs "r40_floorB/source,r39_fixF/removal,r40_floorB/removal" `
     --labels "SOURCE|R39 (looks good)|R40 lower lid over the iris" --track limbus_track_tv.json --name r40_floorB_eyes --status r40_floorB --per-sheet 40
# CPU reference for the new track: eyelid_fit.py ... --lower-iris-edge --device cpu (identical)
& $P -B scripts/probe_r40_ellipse.py "961:left,962:left,1050:left,700:left" out.json   # round 41's starting data
```

## 8. Starter prompt for round 41

> Stay on eyes, on `reporter_full`. Read round40.md §2, §3 and §6 first. Round 40 found that
> 933–955 right was the LOWER LID, not foreshortening, and fixed it at its cause
> (`eyelid_fit.py --lower-iris-edge` + `eye_removal_gpu.py --lid-floor`, run `r40_floorB`:
> 15 eyes back to the full fill, nothing else moved). Ask for / record his verdict on
> `final/r40_floorB_eyes_phone.mp4` first — whatever he raises outranks this.
>
> Then the real off-axis class, which round 40 located but did not build: at a hard sideways
> glance (961–964, 1050–1054, 125 left; the head turn 110–214) her iris is foreshortened and
> the fixed-radius limbus circle sits beside it, so the pass drops an open eye with the softbox
> up. `probe_r40_ellipse.py` separates the glances cleanly (+23–37 % over the best circle at
> b/r ≈ 0.63; ordinary eyes 0 %) except next to a glance, where a narrow ellipse displaced the
> other way wins by 3–8 % — reject that before anything paints. Build the elliptical limbus
> (GPU, gated on changing nothing else: a run with it must reproduce r40_floorB on every eye
> it does not touch), write `smoothed_axes = [b, r]` — the pass already paints ax ≠ ay — and
> look at every eye it moves against r40_floorB before sending. 1050 left is where round 39
> §2a painted a second pupil.
>
> Keep: GENERIC tool, GPU first, exact source outside the weight, p95 and max. Softbox region
> replaced whole by geometry; shading from HER pixels filtered in time with symmetric
> windows; a neighbour's lid never copied from one side at a blink. The scene catchlight stays
> PARKED; the scleral glint only if he asks. No key retuning, no capture requirements.
