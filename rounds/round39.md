# Round 39 handoff — the eye stage on the GPU, bit-exact; round 38's evidence fixed at its causes

**Date:** 2026-09-10. Starts from round 38 (`reporter_full`, the whole 20 s clip, "really,
really close to perfect overall"). Production defaults unchanged — every new behaviour is a
flag; the full command is in §7.
**Shown to the artist:** `reporter_full/final/r39_fixE_eyes.mp4` (SOURCE / R38 / R39 stacked,
12 fps, 1:40; a 31 MB phone copy was sent). **That video has a regression I introduced and
found only afterwards, while writing this handoff: at source 1049–1055 left (a fast glance)
and 848–853 left it paints a displaced iris — grey blotches, a second pupil, dark smears on
the sclera — on frames R38 had left alone.** Cause and fix in §2a; the corrected run is
`r39_fixF`, video `final/r39_fixF_eyes.mp4`.
**Artist verdict on `r39_fixF_eyes.mp4`: *"looks good, commit and hand off."*** r39_fixF is
the round's result; its command is §7.

His one instruction this round, mid-way, when I proposed CPU shortcuts to the eye pass:
*"Port it to the GPU instead of optimizing it on the CPU."* It is in AGENTS.md and memory.

---

## 0. What was built

**The eye stage on the GPU, reproducing the CPU bit for bit.**

| stage | CPU (R38) | GPU (R39) | reproduction on reporter_full |
|---|---:|---:|---|
| `limbus_fit.py --device cuda` (default) | ~3 min | 72 s (120 s cold disk) | every row of `limbus_track.json` identical |
| `eyelid_fit.py --device cuda` (default) | 67 s | 8.7 s | every row of `eyelid_track_w.json` identical |
| `eyelid_fit.py --update` (new, below) | — | 2.0 s | `eyelid_track_wt.json` identical |
| `eye_removal_gpu.py` (the eye pass) | 21 min | 64 s, GPU ~94 % | all 1200 removal + weight PNGs of `r38_full_own` identical, 0 decision differences |

`logs/r39_trackers_gpu.json`, `reporter_full/logs/validate_r39_gpu_vs_r38_full_own.json`.
The whole eye stage is now ~2.5 min per 1200-frame clip instead of ~25.

* **`scripts/eye_removal_gpu.py`** — the eye pass in torch, batched over eye-frames grouped
  by (side, exact window shape) so every border rule is the one-window rule; holds eye
  windows, not frames; cross-frame steps gather across groups. Same CLI as
  `eye_soft_removal.py` (its `build_parser()`); any flag the port does not implement is
  REFUSED (`check_supported`), never half-run. `eye_soft_removal.py` stays as the reference.
  `--dump` writes `diag.npz` (every treated eye's window, lid fits, lash line, weights,
  status) for the audits.
* **`scripts/eye_gpu.py`** — the primitives, each against cv2/numpy (`--selftest`, all exact
  where the trackers need exact; §3). **`scripts/track_gpu.py`** — the trackers' fields and
  sampling. The small robust fits (circle, centre, parabola, ~60 points an eye) run
  afterwards in the CPU scripts' own numpy functions on the GPU's exact samples (§3 why).
* **`scripts/validate_eye_gpu.py --ref <run> --test <run>`** — compares two runs from what
  they WROTE: PNGs, canonical npz, per-eye records, decisions.

**Failure classes found automatically over all 2400 eyes, then looked at.**
* **`scripts/audit_r39_classes.py --name <run>`** — reads the run's output (+ its `--dump`):
  `painted_skin` (fill over source brighter than halfway between her iris and her lid skin,
  outside the softbox region), `limbus_offset` (round 35's independent level-crossing
  estimator), `softbox_left_{treated,untreated_open,untreated_closed}` (corneal specular
  found in the RESULT alone). `logs/r39_classes_<run>.json`.
* **`scripts/review_r39_classes.py`** — corner-stabilised crops per class, SRC / RESULT
  (/ `--ref` RESULT) / overlay of what the pass used (limbus cyan, lids yellow/orange, lash
  line green, fill magenta, softbox red), into `composite/r39_review/`.

**Three fixes, one per cause** (§2):
* **`--lid-claim 0.8`** — an eye with NO lash line whose fitted upper lid claims more than
  0.8 limbus radii of open iris above the centre gets the softbox region only.
* **`scripts/limbus_trust.py`** → `relight/limbus_track_tv.json`, then
  **`eyelid_fit.py --update eyelid_track_w.json`** → `relight/eyelid_track_wtv.json`: any
  frame whose limbus centre SE exceeds 2.5× its eye's local (1 s) median is a candidate for a
  BRIDGE from the trusted frames either side (their offset from the eye corners,
  interpolated; gaps up to 0.25 s) — **and the image decides** (§2a): each run bridged from
  the same two frames keeps the bridge only if the fit's own objective (Daugman's mean
  radial derivative at the shot radius, lid-free angles; `track_gpu.edge_support`) is at
  least as high at the bridged centres as at the fitted ones, by the run's median. 53 eyes
  bridged (43 L, 10 R), 35 refused. `--update` refits the lids of exactly the moved eyes with
  the curvature they already had and copies the rest. (`--no-verify` is the first version,
  88 bridges, which wrote `limbus_track_t.json` / `eyelid_track_wt.json` for r39_fixE.)
* **`--skin-gate 6 --skin-gate-low 3`** — an eye whose fill still covers ≥ 6 px of her skin
  or sclera gets the softbox region only; the gate spreads in time to neighbours still
  showing ≥ 3 px (hysteresis), so a run is decided once.

## 1. Results (`r39_fixF` vs `r38_full_own`, both on the GPU pass)

Exact source outside the weight (0 px). 174 frames identical, the rest differ inside the
weight (p95 of per-frame max 58 levels, max 194 — the fixed eyes). Untouched eyes move by
about 1 level because the canonical is re-pooled from the corrected tracks.
`logs/diff_r39_fixF_vs_r38.json`, `logs/r39_eyes_audit_F.json` (one audit, one lid track,
all three runs), `logs/r39_classes_r39_fixF.json`.

| `audit_r37_eyes.py`, all 2107 tracked eyes | r38_full_own | r39_fixE (sent first) | **r39_fixF** |
|---|---:|---:|---:|
| treated eyes | 2013 | 2024 | 2016 |
| fragment eyes / px | 718 / 1167 | 726 / 1183 | 722 / 1173 |
| region standout median / eyes > 0.05 | +0.012 / 77 | +0.012 / 75 | +0.012 / 77 |
| upper-level p95 / max abs | 0.055 / **0.380** | 0.039 / 0.122 | 0.039 / **0.118** |
| lower-level p95 / max abs | 0.024 / 0.226 | 0.024 / 0.216 | 0.024 / **0.047** |
| lash eyes / px (brightened px) | 232 / 685 (155) | 206 / 455 (40) | 207 / 452 (40) |
| flicker p95 / max | 0.043 / 0.321 | 0.036 / 0.342 | 0.035 / 0.328 |
| departure p95 / max | 0.049 / 0.324 | 0.035 / 0.266 | 0.036 / **0.152** |

| `audit_r39_classes.py`, all 2400 eyes | R38 | r39_fixE | **r39_fixF** |
|---|---:|---:|---:|
| painted skin (≥ 6 px) | **49 eyes** | 0 | **0** |
| independent limbus offset median / p95 / max | 0.57 / 2.47 / 22.5 px | 0.56 / 2.15 / 17.0 | 0.56 / 2.28 / 18.0 |
| softbox left, treated eye | 7 | 6 | 4 (82, 568, 852, 860 L) |
| softbox left, untreated OPEN eye | 21 | 21 | 21 (L: 225, 295, 1051; R: 121–213 far eye, 379, 1101–1102, 1149) |
| softbox left, untreated closed eye | 20 | 20 | 20 |

**The class audit misses standing softboxes**: 1050 and 1052–1054 L are untreated with the
softbox plainly visible (R38 and R39 alike, §2a) and only 1051 is listed — its specular
detector does not find a motion-blurred highlight. Fix the detector before trusting that row.

Flicker: fixE's new maximum (960/1049 L) was the regression; fixF's worst pairs are 441/443
L (R38's own worst) and 492/499 L, 1052 R — the edges of kept bridges. Looked at in r39_fixF
(`composite/r39_review/fixF_flicker_*`): clean and continuous, and better than R38 at 493 and
499; the score there is her eye moving.

## 2. The artist's evidence frames, and every other eye of the same class

* **e1, 137 L (whole disc over a squint)** and the same class at 293, 301–304, 617,
  624–626, 651–653, 859, 1099 L and 302 R: no lash line, while the fitted upper lid — on
  the brow crease, or bridged from frames where it was — claimed > 0.8 radii of open iris.
  The two contradict each other. Of 128 treated eyes with no lash line exactly these 15
  exceed 0.8 and all 15 painted lid skin: no false positive. Now softbox-only. **Looked at.**
* **e2, 243 L (double iris)**: the limbus fit slid on 243–245 and SAID SO (centre SE
  0.24–0.31 px vs 0.04 either side). Round 37's one-frame gate then took the slid frames as
  neighbours and dragged the good ones (241, 242, 246, 247) toward them. Re-derived from
  the trusted frames: a single, clean iris on 241–247. **Looked at.**
* **e3, 304 L (iris over the lid)**: the same contradiction as e1 (lash line cut nothing,
  lid claims the crease). Softbox-only now. **Looked at.**
* **e4, off-axis gaze** (933–955 R, plus lower-lid cases): the canonical iris is a CIRCLE;
  her foreshortened iris is not, so a correctly placed canonical still spills onto the
  sclera. The skin gate routes those eyes to softbox-only — a fallback, not a fix.
* **e5, fast motion**: not addressed. 1050–1054 L keep the softbox, as in R38 (§2a).
* **e0, scleral glint**: out of scope, unchanged.

## 2a. The regression in the video I sent, and the fix (r39_fixE → r39_fixF)

Found while writing this handoff: the eye audit's worst new flicker pairs (960, 1049, 492,
499 L; 1052 R) all sat at the edges of runs `limbus_trust.py` had bridged. Looked at, every
one of the 88 bridged eyes, R39 over R38 (`composite/r39_review/rederived_*`):
* **Right**: 243–245, 442–443, 493–499, 547 L and 717–718, 1050–1052, 866 R — R38's double
  irises and standing softboxes, gone.
* **Wrong**: 1049–1055 L (a fast glance: grey blotched disc, second pupil, dark smears on the
  sclera — nightmare fuel), 849 and 853 L (a dark blob, a dark bar under the lid). R38 had
  left all of them alone.

At a glance the fits were uncertain because of motion blur but RIGHT: her eye moved against
its corners, which is exactly what a bridge assumes does not happen. At 243 the opposite:
the fit sat on the wrong edge. **The SE says a fit is uncertain; only the image says which
answer is wrong.** Scoring both centres with the fit's own objective separates them cleanly
(`logs/r39_limbus_trust_verdict.json`): run median gain +0.030 to +0.076 on 243–245,
442–443, 493–499; −0.012 on 848–853, −0.053 on 1049–1051, −0.046 on 1053–1055. Decided per
run because single frames inside the glance can prefer either circle (1051 L alone would
have kept its bridge). Looked at after the fix (`composite/r39_review/fixF_changed_*`):
every refused frame is back to R38's result, every kept fix is still there.

The refusal also returns 21–26 L (a squint, where the bridge had moved the painted pupil and
neither placement is clearly right) and 961–963 L (the bridge had let 961 be treated) to
R38's result.

## 3. Exactness — what it took to reproduce the CPU on the GPU (do not re-derive)

The gate was bit-for-bit reproduction, and every primitive had to be cv2's or numpy's
actual arithmetic on this machine (opencv 4.13 AVX2 build, numpy 2.4 + OpenBLAS). Pinned by
probe, held by `eye_gpu.py --selftest`:
* **`cv2.remap`** quantises maps to 1/32 px and uses a float weight table. Linear sums
  `((s00·w00 + s01·w01) + s10·w10) + s11·w11`; nearest uses `cvRound`. **Cubic**
  (INTER_CUBIC, a = −0.75 table): 16 taps summed row by row inside the image, one at a
  time at its border. `CubicSampler` does it with ONE gather of a precomputed 4×4
  neighbourhood — 16 separate 64-bit-index gathers made the GPU no faster than numpy.
* **`cv2.GaussianBlur` 7×7 float**: FMA chains on 8-wide vectors, plain multiply-add on the
  row's scalar tail, rows then (symmetric-pair) columns. Also the `(0, 0), 0.8` reflect-101
  kernel of the eyelid specular map.
* **`cv2.Sobel` 3×3**: `(a+c)+2b` on vectors, `(a+2b)+c` on tail pairs, `(a+c)+2b` on a
  trailing odd pixel; the derivative tap one subtraction.
* **`rgb @ LUMA`** (OpenBLAS gemv per row): FMA chain from x0 on 8-blocks, from x1 on a
  4-block and a last pixel, plain on a 2-block.
* **`cv2.resize` INTER_AREA 4×**: block rows `((a0+a1)+a2)+a3`, rows added in order.
* **numpy**: float32 `mean` is an 8-accumulator pairwise sum; `percentile` lerps in float32
  because the weight is a weak scalar (NEP 50: a Python float meets float32 AS float32 —
  compares included); `median` is the mean of the two middle values in the array's dtype.
  sRGB of half-float frames comes from a 65536-entry table built by numpy itself.
* **torch on CUDA divides by a CPU scalar via a reciprocal** — not IEEE division. Divide by a
  0-d DEVICE tensor (`eye_gpu.div` / `dev_scalar`).
* **Rank-deficient IRLS**: when an iteration leaves ONE weighted point, LAPACK meets an
  exactly zero pivot and numpy raises, so the CPU fit stops; a GPU determinant of the same
  matrix is tiny but not zero and the fit wandered to a 3 px circle (135 R). With TWO
  points (rank 2 in a 3-parameter fit) numpy carries on and the step along the null
  direction is a ratio of rounding residues — no reimplementation reproduces it. So the
  robust fits run in numpy on the GPU's exact samples: a few seconds of CPU per clip, and
  the tracks are identical. (Commit 198a460 still describes the earlier tolerance; d1ff0c9
  supersedes it.)

## 4. Tried this round and set aside

| try | why |
|---|---|
| optimise the eye pass on the CPU (vectorise the hot spots) | the artist: *"Port it to the GPU instead"* |
| `--refine-limbus` (re-place each eye by matching the canonical over ±8 px) | right on the big slides, but moved 173 eyes and put correctly placed ones 4 px off (721–722 L, 703–717 R); the independent estimator disagrees with it exactly on 243–245. Kept as a flag, not used |
| full eyelid refit after `limbus_trust.py` | re-pooled the 2.5 s curvature and so moved the lids on 1333 eyes; `--update` refits only the 88 |
| per-frame skin gate | 933–955 R toggled every few frames — a flicker of its own; hysteresis fixed it |
| `--limbus-trust` inside the pass | re-derives the centre after the lids were already fitted around the slid one; the pre-pass (`limbus_trust.py` + `eyelid_fit --update`) is what the runs use |
| bridging every high-SE run (`limbus_trust.py --no-verify`, r39_fixE) | a high SE is motion blur as often as a slide; at a glance the bridge is the wrong answer (§2a) |
| batched robust fits on the GPU | rank-deficient steps (§3); numpy fits on exact samples instead |

## 5. Metrics and facts worth not re-deriving

* Profile of the CPU pass (200 frames, 181 s): 34 % converting whole frames to sRGB to crop
  34 px review tiles, 31 % the atlas registration search (~1M small remaps), 10 % unpacking
  whole frames to read 40 px windows. The eye maths itself was ~6 %.
* `validate_eye_gpu.py` on r39_gpu vs r38_full_own: canonical eyes equal to 3e-8, four
  record numbers differ in the last printed digit; everything written is identical.
* Remaining approximate primitives (`gaussian_blur`, `sobel3`, chamfer DT, box blur: within
  5e-7 of cv2) are used only where the eye pass is still verified end to end.
* Limbus trust: 88 candidate eyes (76 L, 12 R); the image keeps 53 bridges and refuses 35
  (`logs/r39_limbus_trust_verdict.json`, per run). The largest SE ratios are 8.8× (493–499 L,
  a head turn, bridge kept) and 8.1× (244 L, kept).
* The largest independent limbus offsets in r39_fixF are untreated eyes (129 L `no_eye_frame`,
  852 L) and 851 L/R; look at a frame before believing a max.
* `validate_eye_gpu.py --out NAME` writes NAME relative to the working directory, not the
  shot's logs — pass `$F/logs/NAME`.
* The r39_fixF pass took 478 s, not 64 s: another project's job held the GPU at 98 %. Check
  `nvidia-smi` before reading a time.

## 6. Still open — the round 40 list

1. Nothing he has pointed at: r39_fixF "looks good". Whatever he raises next outranks this list.
2. **Off-axis gaze (e4)**: the canonical is painted as a circle. A foreshortened iris needs
   the canonical projected as an ellipse (the eye's own limbus shape — fit an ellipse to the
   same edge points where the arc allows, or derive the squash from head pose and gaze).
   Until then the skin gate hands those eyes the softbox region only.
3. **Fast motion (e5)**: 1050–1054 L keep the softbox (as R38): during the glance the eye is
   dropped (`no_eye_frame`). The fits there are right (§2a) — the gates downstream are what
   refuse them. Also `detect_anomaly` dropping eyes it does not need to, and 961–963 L.
4. **Surviving softbox on open eyes**: 225, 295, 1051 L; the far eye near profile 121–213 R
   (tiny glints the pass skips); treated eyes 82, 568, 852, 860 L. Look at each. And fix the
   class audit's specular detector, which misses motion-blurred highlights (§1).
5. The generic canonical's softness and right-eye band (R38 §6.7).
6. Scleral glint (e0) — only if he asks.
7. Speed beyond the eye stage: keying is now the slowest stage (22.5 min per 1200 frames,
   ~1.1 s/frame with CPU pre/post) — GPU first applies there too when it is touched.

## 7. Reproduce

```powershell
$P = "dependencies/CorridorKey/.venv/Scripts/python.exe"
$F = "music_videos/reporter_test/shots/reporter_full"
# (frames, key, face track as round38.md §7)
& $P -B scripts/limbus_fit.py --shot $F --count 1200                      # GPU, = limbus_track.json
& $P -B scripts/eyelid_fit.py --shot $F --curvature-window 2.5 --out eyelid_track_w.json
& $P -B scripts/limbus_trust.py --shot $F --out limbus_track_tv.json   # image-verified bridges
& $P -B scripts/eyelid_fit.py --shot $F --track limbus_track_tv.json --update eyelid_track_w.json --out eyelid_track_wtv.json
$base = @('--shot', $F, '--count', '1200', '--canonical-votes-frac', '0.44', '--fill-residual-gate', '8',
  '--fill-iris', '0.99', '--grain', '0', '--pupil-target', '0.34', '--pupil-colour',
  '--limbus-gate', '1.0', '--lid-temporal', '--lid-line', 'parabola', '--lid-line-window', '2',
  '--specular-reach', '2.0', '--no-lash-exempt', '--region-rows', '--top-shading', '2.5',
  '--shading-sides', 'all', '--shading-temporal', '2', '--gain-ring', 'lower', '--gain-temporal',
  '--level-window', '2')
# R38 reproduced on the GPU (the gate):
& $P -B scripts/eye_removal_gpu.py @base --name r39_gpu --eyelid-track eyelid_track_w.json
& $P -B scripts/validate_eye_gpu.py --shot $F --ref r38_full_own --test r39_gpu
# R39 (r39_fixF, the corrected run):
& $P -B scripts/eye_removal_gpu.py @base --name r39_fixF --dump --limbus-track limbus_track_tv.json `
     --eyelid-track eyelid_track_wtv.json --lid-claim 0.8 --skin-gate 6 --skin-gate-low 3
& $P -B scripts/validate_eye_gpu.py --shot $F --ref r38_full_own --test r39_fixF --out "$F/logs/diff_r39_fixF_vs_r38.json"
& $P -B scripts/audit_r39_classes.py --shot $F --name r39_fixF --limbus-track limbus_track_tv.json
& $P -B scripts/audit_r37_eyes.py --shot $F --names "r38_full_own,r39_fixE,r39_fixF" --eyelid-track eyelid_track_wtv.json --out r39_eyes_audit_F.json
& $P -B scripts/review_r39_classes.py --shot $F --name r39_fixF --ref r38_full_own --items "1049:left,1050:left" --neighbours 0 --cols 12 --out look
& $P -B scripts/review_r35_eyes.py --shot $F --dirs "r39_fixF/source,r38_full_own/removal,r39_fixF/removal" `
     --labels "SOURCE|R38 (shown last round)|R39" --track limbus_track_tv.json --name r39_fixF_eyes --status r39_fixF --per-sheet 40
# r39_fixE (the first video, with the regression): limbus_trust.py --no-verify --out limbus_track_t.json,
#   eyelid --update ... --out eyelid_track_wt.json, and the same pass with those two tracks.
# CPU references: limbus_fit/eyelid_fit --device cpu; eye_soft_removal.py with the same @base.
& $P -B scripts/eye_gpu.py --selftest
```

## 8. Starter prompt for round 40

> Stay on eyes, on `reporter_full`. Read round39.md §1, §2, §2a and §6 first. The whole eye
> stage now runs on the GPU in ~2.5 min per 1200 frames and reproduces the CPU bit for bit
> (§0, §3) — iterate with `eye_removal_gpu.py`, never the CPU script, and keep `--selftest`
> and `validate_eye_gpu.py` green when touching a primitive.
>
> The artist's verdict on `reporter_full/final/r39_fixF_eyes.mp4` (SOURCE / R38 / R39):
> *"looks good"*. Round 38's evidence frames 137, 243 and 304 L are fixed at their causes, and
> 49 eyes that painted iris over her skin are now 0. `r39_fixF` (round39.md §7) is the
> baseline: gate every change on `validate_eye_gpu.py --ref r39_fixF` and put the two runs in
> the same tile. If he finds new failures, take his frames as evidence (as in
> `rounds/evidence/round38/`), find every eye of that class with `audit_r39_classes.py`, and
> look at them with `review_r39_classes.py` — **and look at every eye a change touches,
> against the previous run, BEFORE sending a video**; that is how §2a should have been caught.
>
> Then, in order: **off-axis gaze** — paint the canonical as the eye's own foreshortened
> limbus (an ellipse), not a circle, so the skin gate stops being the answer at 933–955 R;
> **fast motion** (1050–1054 L keep the softbox: the fits are right, the gates refuse them)
> and `detect_anomaly`; the **surviving softbox on open eyes** (§6.4). Put frames in front of
> him early.
>
> Keep: the tool is GENERIC — no borrowing from this clip's earlier runs, no per-clip
> constants that do not scale. GPU first. Exact source outside the weight. Report p95 and
> max. Softbox region replaced whole by geometry; shading from HER pixels filtered in time
> with symmetric windows; a neighbour's lid never copied from one side at a blink. The scene
> catchlight stays PARKED; the scleral glint only if he asks. No key retuning, no capture
> requirements.
