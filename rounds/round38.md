# Round 38 handoff — the whole 20 s clip: "really, really close to perfect", with evidence

**Date:** 2026-09-10. Starts from round 37 (accepted on source 700–849). Production defaults
unchanged — every new behaviour is behind a flag; the full command is in §7.
**Artist verdict on `final/r38_full_own_eyes.mp4`:** *"Mostly solid results, but there was
some truly nightmare fuel here ... It does happen here and there in the frames across the
full 1:40. It's really, really close to perfect overall though."* Then: *"Write these things
down, and track these frames as evidence, for tackling in a fresh round, where we'll also
move to the GPU for faster iteration time."*

**Every evidence frame is in `rounds/evidence/round38/`** — his screenshots, stacked
SRC / result crops of each defect with its neighbours, and `evidence.json` with what the
pipeline recorded on each one. Read that before touching code.

Two standing instructions from him this round, both now in AGENTS.md:
* **This is a GENERIC tool** — *"it can accept any kind of input video."* Nothing may be
  tuned to, or borrowed from, this clip's earlier runs in a way that another input could
  not do for itself (§4: that is why `--canonical-from` is diagnostic only).
* **GPU first** — *"Never run stuff on the CPU when it can be avoided."* The eye pass ran at
  1 % GPU / 48 W, single-threaded numpy, one of 32 threads (§5).

---

## 0. What was built

* **`reporter_full`**, a new shot: the same plate, `selection` 0 / 1200, reporter_a's key
  thresholds copied (not recalibrated). reporter_a stays the anchor for every earlier round.
  Extraction reproduces reporter_a: **source 700–849 frames and keyed EXRs are bit-identical
  to reporter_a's 0–149 (150/150 each).** Face found on 1200/1200 frames.
* **`scripts/frame_store.py`**: the eye passes hold every frame in RAM; as float32 that was
  ~30 GB at 1200 frames. Keyed EXRs are HALF float, so they are held as float16 and the round
  trip is ASSERTED exact per frame; atlas PNGs are held as their 8 bits. Frames come back
  read-only. **Re-running R37w's exact command reproduces R37w bit for bit (150/150 removal
  and weight, both canonicals).**
* **`--fill-residual-gate K`** (`eye_soft_removal.py`): a frame whose own robust residual
  against the canonical is > K × the shot's gets only the softbox's own region, like a frame
  with no upper bound. Found at source 827 left (§3). On reporter_a: median 1.0×, squints
  ~3.4×, a clean gap 6.0×–11.3×; K = 8.
* **`--canonical-votes-frac F`**: `--canonical-votes 60` was 44 % of reporter_a's ~138
  pooled frames. Over 1200 frames the top of the iris gets 50–170 votes and would pass 60
  unrebuilt — silently switching round 37's symmetric rebuild off. F = 0.44 scales it (462 /
  454 votes here), lo:hi kept at 10:60.
* **`eyelid_fit.py --curvature-window SECONDS`**: one lid curvature per eye per shot is wrong
  over 20 s. The median free upper-lid curvature per 100 frames runs 0.003–0.023 as her head
  and gaze move (and goes negative where the fits latch onto the wrong edge); the shot median
  (0.011 / 0.013) was 2.3–3.4× reporter_a's 0.0048 / 0.0038 on the accepted stretch.
  Pooled over a shifted 2.5 s window, **anatomical sign only** (upper lids arch up): at
  source 800–899 left half the free fits curve the wrong way and an unsigned windowed median
  came out negative. Upper-lid column residual p95 0.93 / 0.71 px vs 1.07 / 1.03 with the shot
  constant, and more frames fit (858 / 894 vs 847 / 890). `eye_soft_removal.py` uses the
  per-frame value for the lash line.
* **`--canonical-from <run>`** — DIAGNOSTIC ONLY (§4). Paints another run's canonical, at the
  same size in pixels. With the gate, it reproduces R37w bit-exactly on all 150 reporter_a
  frames without the 150-frame inpaint atlas, which proves the rest of the chain is R37w's.
* **Review**: `review_r35_eyes.py --status <run>` names each untreated eye on the result row
  (orange); `--fps-exact`. `verify_r34.py` reports reporter_a's fallback fixtures as absent on
  other shots instead of crashing. `audit_r37_eyes.py --eyelid-track`. `limbus_fit.py` no
  longer crashes after writing when a shot has a bridged eye. New `audit_r38_sclera.py`.

## 1. Results (`r38_full_own`, the generic run)

Verification: 1200 frames, **exact source outside the weight**, 2013 treated eyes all darker
than source at the highlight. reporter_a's 300 fallbacks / 304 hashes are not fixtures of
this shot. `production_enabled` and `artist_accepted` are false.

| `audit_r37_eyes.py`, all 2107 tracked eyes | r38_full_own | r38_full (R37w iris) | R37w (150 fr) |
|---|---:|---:|---:|
| treated eyes | 2013 | 2014 | 273 |
| fragment eyes / px | 718 / 1167 | 710 / 1160 | 94 / 130 |
| region standout median / eyes > 0.05 | +0.012 / **77** | +0.002 / 8 | +0.006 / 0 |
| upper-level p95 / max abs | 0.055 / 0.380 | 0.058 / 0.361 | 0.045 |
| lower-level p95 / max abs | 0.024 / 0.226 | 0.026 / 0.228 | 0.020 / 0.031 |
| lash eyes / px (brightened px) | 232 / 685 (155) | 167 / 529 (151) | — / 63 (17) |
| departure p95 / max | 0.049 / 0.324 | 0.050 / 0.471 | 0.037 / 0.095 |

The generic canonical (her own pixels over 1050 / 1032 frames) is rounder than R37w's on
the left eye but SOFTER (a median over a moving head), and the right one has a lighter
straight-edged band above the pupil (`composite/r38_full_own/canonical_right.png`). Region
standout over 0.05 on 77 eyes against 8 with R37w's iris is that material.

**The accepted stretch is not reproduced inside the full clip**: at source 700–849 every
treated frame differs from R37w inside the weight (p95 of per-frame max 36–38 levels, max
76–79), in BOTH full runs — so it is the tracks, sigma and shading re-derived over 1200
frames, not the iris material. Nothing differs outside the weight.

Untreated eyes: 150–171 per eye, almost all blinks (aperture < 0.8). On an OPEN eye, left
only: 124–125, 295, 494–499, 961–964, 1050–1054 (§2 e4, e5).

Timing (RTX 5080 box, 9950X3D): key 22.5 min for 1200 frames (GPU, with CPU pre/post, and
sharing the CPU with other jobs); face track a few minutes; limbus 3 min; eyelid 1 min;
**eye removal 21 min per run** (two ran at once). GPU utilisation sampled during the eye
pass: **1 %, 48 W**.

## 2. Everything the artist caught, and what it was (`evidence.json` has the numbers)

All three on her LEFT eye (left tile), all in the head-turn stretch (yaw to −47° at 100–250).
* **e1, 137: "the full iris and pupil ... none of it was behind the eyelid".** Limbus 17 %
  visible (squint), but the upper-lid fit sits 20 px / 1.65 radii above the centre — the brow
  crease, inside `LID_BAND`'s 1.70 — and the lash line cut nothing. Whole disc painted. The
  residual gate saw only 3.1×, an ordinary squint.
* **e2, 243: "double-iris".** The limbus slid over SEVEN frames (241–247, centre SE 0.30 px
  vs ~0.04); the one-frame gate fired on all seven and repaired each from neighbours that
  had slid too. Round 37 §6 predicted exactly this.
* **e3, 304: "iris went above the eyelid".** Upper lid unfitted 302–306; on 303–304 a lash
  line existed and cut 0 px, so the full disc went over the lid. 305–306 are fine.
* **e0, the scleral glint**: a second capture-room reflection on the WHITE of the eye,
  outside the construction by design (fill stops at 0.99 radii). `audit_r38_sclera.py` finds
  candidates on 468 left-eye frames; looked at, a real speck on the temporal sclera ~2 radii
  out on many frames, plus inner-corner specks that are a different surface — an upper bound.

Found by looking before he did: **e4** sideways gaze (lid opening collapses on an open eye →
softbox pops for 4–6 frames; softbox straddling the limbus leaves a light crescent past
rho 1.02 on treated frames); **e5** fast motion (limbus lands off the iris or abstains; round
32's `detect_anomaly` "no reflection" also drops the eye from the whole pass); **e6** 827
blink edge — fixed by the gate.

## 3. The one fix made this round, and why it had to be

**827 left, the frame before a blink**: the upper-lid fit on the crease let the fill paint a
black pupil onto the closing lid. R37w escaped it only because its NCC veto fired BY
ACCIDENT (−0.25 against the ProPainter-pooled atlas; 0.54 against her own pixels). The image
says it plainly — the fitted opening does not look like her iris, 13.5× the shot residual —
so `--fill-residual-gate 8` routes such frames to softbox-only. It does NOT catch e1/e3: there
the opening's residual is squint-normal, because the wrong part is the BOUND, not the content.

## 4. Tried this round and set aside

| try | why |
|---|---|
| paint R37w's canonical over the full clip (`--canonical-from`) | only possible because this clip IS reporter_a's footage; another input has no accepted run to borrow from. Kept as a diagnostic — it separates "new material" from "method meets new footage" |
| one lid curvature for the whole clip | 2–3× wrong on the accepted stretch (§0) |
| unsigned windowed curvature | negative (inside-out lids) at 800–899 left |
| absolute `--canonical-votes 60` on 1200 frames | switches the symmetric rebuild off (§0) |

## 5. Metrics and facts worth not re-deriving

* **The GPU is idle during the eye pass** (1 %, 48 W) — it is single-threaded numpy. Keying is
  the only GPU stage and still ~1.1 s/frame with its CPU pre/post. At today's speed 100 clips
  × 3 min × 24 fps is ~5 days of eye pass alone; the artist wants this moved to the GPU first.
* reporter_a's committed `eyelid_track.json` was fitted against the round-35 limbus track
  (written 00:42, the limbus track 09:15): re-running eyelid_fit on reporter_a today gives
  117/116 fitted eyes and right-upper curvature 0.00377 against the file's 116/116 and
  0.00384. R37w used the file as it is.
* Limbus shot radius over 1200 frames: 12.235 / 12.267 px (reporter_a 12.303 / 12.271).
* `removal.json` hashes its own code at the END of a run. `r38_full` (diagnostic) was running
  while `--canonical-votes-frac` and the per-frame lash-line curvature were added, so its
  recorded `code_sha256` is of the later file. For its arguments the two are the same
  behaviour (the flag was off and its lid track carries no per-frame curvature), but the hash
  is not the code that ran. `r38_full_own` started after the last edit.
* Free-fit limbus radius has a p5 of 7.7 px on the left eye — some frames fit the PUPIL edge.
  The shot median is robust to it; a per-frame consumer would not be.

## 6. Still open — the round 39 list, in the artist's order of pain

1. **Upper bound far above the real lid** (e1 137, e3 303–304): the lid fit on the brow crease
   or a lash line that cuts nothing, on squints and head turns. Cross-check the bound against
   what the IMAGE says is visible: at 137 the limbus is 17 % visible while the lids claim an
   opening of 2.75 radii. `LID_BAND` allows 1.70 radii above the centre, which is the crease.
2. **Multi-frame limbus slide** (e2 241–247): the gate is single-frame by design. Needs a
   slide model, e.g. the offset from the eye corners over a window, or the fit's SE.
3. **Off-axis gaze** (e4): lid opening collapse on an open eye; softbox beyond rho 1.02.
4. **Fast motion** (e5), and `detect_anomaly` dropping eyes it does not need to.
5. **Scleral glint** (e0) — not in the construction at all.
6. **Speed**: port the eye pass to torch/CUDA; hold eye windows, not frames; parallelise.
7. The generic canonical's softness and right-eye band.

## 7. Reproduce

```powershell
$P = "dependencies/CorridorKey/.venv/Scripts/python.exe"
$F = "music_videos/reporter_test/shots/reporter_full"
$ff = "dependencies/ffmpeg/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe"
& $ff -hide_banner -y -i "$F/source/plate.mp4" -vf "scale=in_range=limited:out_range=full:in_color_matrix=bt709" `
      -fps_mode passthrough -pix_fmt rgb48be -start_number 0 "$F/source/frames/%06d.png"
& $P -B scripts/key_plate.py --shot $F
& dependencies/facetrack/.venv/Scripts/python.exe -B scripts/face_track.py --shot $F
& $P -B scripts/limbus_fit.py --shot $F --count 1200
& $P -B scripts/eyelid_fit.py --shot $F --curvature-window 2.5 --out eyelid_track_w.json
& $P -B scripts/eye_soft_removal.py --shot $F --name r38_full_own --count 1200 `
     --eyelid-track eyelid_track_w.json --canonical-votes-frac 0.44 --fill-residual-gate 8 `
     --fill-iris 0.99 --grain 0 --pupil-target 0.34 --pupil-colour `
     --limbus-gate 1.0 --lid-temporal --lid-line parabola --lid-line-window 2 `
     --specular-reach 2.0 --no-lash-exempt --region-rows `
     --top-shading 2.5 --shading-sides all --shading-temporal 2 `
     --gain-ring lower --gain-temporal --level-window 2
& $P -B scripts/verify_r34.py   --shot $F --name r38_full_own
& $P -B scripts/audit_r37_eyes.py --shot $F --names "r38_full_own,r38_full" --eyelid-track eyelid_track_w.json --out r38_eyes_audit.json
& $P -B scripts/audit_r38_sclera.py --shot $F --eyelid-track eyelid_track_w.json
& $P -B scripts/review_r35_eyes.py --shot $F --dirs "r38_full_own/source,r38_full_own/removal" `
     --labels "SOURCE|R38 full clip (generic: iris built from this clip)" --name r38_full_own_eyes --status r38_full_own --per-sheet 40

# the diagnostic: R37w's iris, shot-constant lid track (eyelid_track.json), same flags otherwise
#   ... --name r38_full --canonical-from music_videos/reporter_test/shots/reporter_a/composite/r37w --canonical-votes 60
# and the proof it is R37w's chain: on reporter_a, --canonical-from .../r37w --fill-residual-gate 8
#   (no --atlas-frames) reproduces composite/r37w bit-exactly, 150/150.
```

Reviews (display-sRGB CRF 12, 12 fps, 1:40): `reporter_full/final/r38_full_own_eyes.mp4`
(shown to him), `reporter_full/final/r38_full_eyes.mp4` (R37w iris, diagnostic).

## 8. Starter prompt for round 39

> Stay on eyes, on `reporter_full` (the whole 20 s clip). Read round38.md §2 and §6 and
> `rounds/evidence/round38/evidence.json` first — the artist's evidence frames are 137, 243
> and 304 on her LEFT eye, each with a diagnosis from the pipeline's own records. His verdict:
> *"really, really close to perfect overall"*, with those failures *"here and there across the
> full 1:40"*.
>
> **Two things this round, in this order.**
>
> 1. **Move the eye pass to the GPU** — his instruction: *"Never run stuff on the CPU when it
>    can be avoided."* The 5080 sat at 1 % for 21 minutes per run. Port `eye_soft_removal.py`'s
>    per-frame maths (and `limbus_fit.py`, `eyelid_fit.py`) to torch on CUDA, hold eye windows
>    instead of frames, and batch frames. **Gate it on bit-for-bit or tolerance-documented
>    reproduction of `reporter_full/composite/r38_full_own`** before changing any behaviour:
>    the speed-up must not move a pixel the artist has seen, or say exactly which and why.
> 2. **Then the failures, with the faster loop**: the upper bound far above the real lid
>    (e1, e3 — cross-check the lid/lash bound against how much limbus the image shows), the
>    multi-frame limbus slide (e2), off-axis gaze (e4), fast motion (e5). Find ALL frames of
>    each class automatically over the 2400 eyes, not just the three he screenshotted, and
>    put them in front of him early.
>
> Keep: the tool is GENERIC — no borrowing from this clip's earlier runs, no per-clip
> constants that do not scale (vote counts, shot-wide curvature). The scleral glint (e0) is
> logged, not yet in scope unless he says so. The softbox region is replaced whole by
> geometry; shading from HER pixels, filtered in time with symmetric windows; a neighbour's
> lid is never copied from one side at a blink. Exact source outside the weight. Report p95
> and max. The scene catchlight stays PARKED. No key retuning, no capture requirements.
