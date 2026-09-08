# CLAUDE.md

## Read the latest round first

**Before doing anything else in this repo, read the most recent file in `rounds/`.**

```powershell
Get-ChildItem rounds/round*.md | Sort-Object Name | Select-Object -Last 1
```

Each round is the handoff from the previous working session: what was delivered and verified,
what is genuinely broken, decisions the artist has already made (do not re-litigate them), and a
starter prompt for the round you are about to do. The latest round supersedes this file and the
README wherever they disagree.

At the end of a working session, write the next `rounds/roundN.md` in the same shape, ending
with a starter prompt for round N+1.

---

## What this project is

A virtual-production pipeline that composites green-screen footage of the artist into Blender
environments, to produce music videos. The stated acceptance bar is **full photorealism** —
"if it doesn't look 100% authentic and like I was literally there, I don't want it anyway" — so
do not present incremental gains as success. Say plainly when an approach has a ceiling below
that bar.

The artist shoots his own performances at **24 fps**, usually in bust/medium framing with no
feet or floor contact in shot.

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

Two Python environments, deliberately separate:

| purpose | interpreter |
|---|---|
| pipeline (torch 2.8+cu128, opencv 4.13, numpy 2.4) | `dependencies/CorridorKey/.venv/Scripts/python.exe` |
| face landmarks (mediapipe 1.x needs opencv-contrib 5.x) | `dependencies/facetrack/.venv/Scripts/python.exe` |

Isolate any dependency that would disturb the pipeline's opencv — every EXR read depends on it.
Use `uv pip install --python <venv>/Scripts/python.exe <pkg>`; there is no `pip` in these venvs.

Blender is **5.2.1 LTS** at `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`.
A Blender Lab MCP add-on bridge may be listening on `127.0.0.1:9876`; `scripts/blender_mcp_client.py`
speaks it for read-only queries. Scene *building* runs under `blender -b --python`.

Hardware: RTX 5080, **16 GB, Blackwell sm_120** — requires CUDA ≥ 12.8 / torch ≥ 2.7. Any repo
pinning older torch will not produce a working kernel; it fails at runtime, not import.

---

## House rules

- **Verify conventions, never assume them.** Axis, handedness and sign errors in this domain are
  invisible in a single frame and ungradeable afterwards. Test several directions, not one — a
  reflection error leaves the diagonal invariant and passes a single-direction check.
  `scripts/validate_sh_convention.py` is the pattern.
- **Measure before and after.** Every claim in `rounds/` is backed by a number from the repo.
  Prefer a control (relight off) over an unanchored figure.
- **State assumptions as assumptions.** The reporter camera match and `E_source` are both
  assumed and labelled so in `shot.json`. Do not let an assumption harden into a fact.
- **Never recalibrate the proof-of-concept shot's keying thresholds.** They are hand-set because
  that plate has a green costume, and they produced the keys behind a validated master.
- Commit with a message that explains *why*, including bugs found and what they would have cost.
- The README carries pipeline detail and the full list of Blender/library traps.
