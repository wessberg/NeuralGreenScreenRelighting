# Round 20 handoff — stay on eyes; explicit transport and joint separation

**Date:** 2026-09-09. Starts from round 19. Production defaults unchanged.
**Delivered:** a frozen observation-based inverse-eye experiment, explicit conic
reflection/refraction and dual-ray lid occlusion, an isolated coupled replacement
transaction, independent Cycles verification, two labelled videos and source checks.
**Not delivered:** a real reporter eye repair, validated real geometry/pupil radiance,
scene-reflection replacement on performer footage, or photorealism.

## 1. Artist steering — latest priority

The artist said: **"i think you should stay on eyes until theyre fixed though"**.
This supersedes round 19's instruction to prioritize instruments. Stay on eyes.
The broad material/instrument task is deferred, not discarded.

Keep observed performer identity, input-agnostic design, unchanged keying and
existing gates, and exact source fallback including original catchlights whenever
reconstruction is unreliable. Tracking plus coupled source removal/scene-light
replacement remains the task. No special capture reference is a requirement.

## 2. Research and implementation boundary

Read `docs/r20-eye-reconstruction.md` and the frozen `docs/r20-eye-protocol.md`.
The published Seeing the World through Your Eyes method motivates fitting iris
texture, corneal pose and reflected radiance jointly. Its code was cloned at
`9005f9a462a4ca5bcaeb1e101008a2e12798c915` into `dependencies/eyenerf`.

**The full neural method was not installed or trained.** Only its pure conic
geometry functions were executed for parity. Its current texture path routes both
sides through the left texture network, regularizes texture rotationally and
minimum-shifts saved previews. Those details need attention before using its output
as same-eye evidence; they are not a verdict on the method's capability.

The new fitter is an independent sparse linear two-layer model, not a port or
deployment of EyeNeRF. It takes observed RGB plus supplied transport, and fits
bounded interior/environment radiance grids. It has no pretrained RGB decoder,
clean texture input or spatial/rotational regularizer. No performer texture was fit
or generated in this round.

## 3. Frozen synthetic separation

36 procedural nonperformer eye frames, 64x64, explicit conic, Snell/Fresnel,
planar interior and independent lid closure. The 30 training frames have 11,183
visible RGB samples; all 36 have 13,181. Frames 5,11,17,23,29,35 are held out
completely. Frame 17 is closed. The scored holdout has 1,094 interior pixels,
including 967 iris and 127 pupil pixels, excluding prescribed boundary bands.

| Mean absolute display RGB error, 0..255 | Source/add-only control | Unknown pupil | Oracle black pupil |
|---|---:|---:|---:|
| Whole interior | 6.61804 | 1.78639 | 0.00000191 |
| Iris only (supplemental verification breakdown) | 1.80230 | 0.59517 | 0.00000216 |
| Coupled estimate with known target reflection | 6.05533 | 1.35250 | 0.00000189 |
| Fit to observed source sum | — | 1.05492 | 0.00000373 |

There are 404/400 active scalar parameters per channel for unknown/oracle pupil.
Final solve times: 36.27/52.49 seconds. **Three channel solves hit the frozen
200-iteration cap**: unknown red, oracle red/blue. They remain flagged unconverged.
No tolerances or iteration limits were retuned. Source has zero channels above one.

Critical limits:

- The generator uses exactly the same bilinear basis as the fitter. This is a
  controlled algebra/transport positive test, not independent footage reconstruction.
- The oracle assumes the exact black pupil and geometry. Never assume a real pupil
  is black. The unknown arm also approximates the pupil discontinuity with a
  continuous grid, so this is not a pure pupil-radiance ablation.
- The interior is emissive and homogeneous behind one optical interface. No rear
  cornea, aqueous/lens interfaces, iris scattering, lashes, variable pupil or real
  eyelid geometry. Changing reflections is not full diffuse iris relighting.
- No uniqueness/uncertainty certificate and no clean-donor-coverage result was
  earned. A static-ray algebraic control changes interior radiance by 0.03364 while
  matching RGB within 5.56e-17; this is not a moving-footage impossibility bound.

## 4. Transport, coupled replacement and verification

`corneal_transport.py` uses explicit native camera intrinsics and supplies material
and reflected-ray coordinates. A lid aperture tests both camera and incident rays.
Camera/world coordinates are x right, y down, z away, lengths millimetres. The
synthetic camera is declared in the protocol, not inherited from the artist's rig.

`render_r20_transport.py` independently builds the surface and traces it in Cycles:
seven cases, constant emitters, 4096 samples, OPTIX, no denoise, persistent data off.
The initial mesh had reversed ring faces: +yaw display error was 10.20470 despite
a plausible frontal picture. Winding was fixed and an assertion added, retaining
the same comparison tolerance. Corrected nonboundary display MAEs are
0.02158..0.04754; p95 <=0.11742. Closed lid is exactly zero. The open cases contain
72..230 camera-visible samples whose incident reflection ray is blocked.

Other checks:

- Conic residual <=8.74e-10; Snell/reflection/unit-vector errors <=3.34e-16.
- Normal-incidence Fresnel 0.0250427961. Four signed image and four signed pose
  controls pass. Author geometry parity: points <=2.51e-11 mm, normals <=1.13e-12.
- 72 saved frame/arm records checked with independent scalar bilinear sampling;
  maximum matrix/scalar difference 8.33e-17.
- Four complete-heldout-RGB substitutions (black/magenta, both arms) leave the
  fitter's entire matrix and RGB inputs identical. This is input-isolation evidence,
  **not a claim of repeated end-to-end solver runs or real-data search isolation**.
- `coupled_eye_relight.py` passes 12 transaction checks. One taper couples removal
  and new reflection; failed flags, partial evidence or nonfinite data preserve
  source exactly. Float RGB and boolean support are required. No production CLI
  or compositor was changed; only known synthetic truth exercises enabled gates.
- **300 exact keyed-eye fallbacks and 304 protected-file hashes pass.** Existing
  real MAE/MAD/donor/context gates and old all-arm exclusions were not changed.

## 5. Reviews, artifacts and development notes

- `final/r20_eyes.mp4`: **144 frames, 24 fps, 960x762, six seconds**. A 36-frame
  synthetic loop repeated four times. Source, two fitted interiors, known target,
  unknown-pupil fit plus known target reflection, and add-only control are labelled.
  The unknown-pupil rim is visibly blurred. Blank frame 17 is the intended closure.
- `final/r20_source_fallback.mp4`: **50 frames, 60000/1001 fps, 640x732**. Real
  source left, exact source fallback right, both eyes. Original catchlights remain.

All 194 decoded thumbnails inspected in nine ordered sheets. Full synthetic boards
5 and 17, real board 35, and Cycles yaw/narrow comparisons were inspected, plus
the native synthetic board 0. No live playback or artist acceptance is claimed.
Maximum mean decoded-frame differences from input PNGs: 0.27571 synthetic and
1.68062 real display levels. Numerical exact-source claims refer to arrays, not
lossy H.264. Review encodes are display sRGB/CRF 12, not HDR delivery.

Reports: `logs/r20_eye_preflight.json`, `r20_eyes.json`,
`r20_cycles_transport.json`, `r20_eye_verification.json`,
`r20_real_fallback.json`, `r20_visual_review.json` under the shot.
Caches and boards: `composite/r20_eyes/`, `composite/r20_source_fallback/`.

The first verification caught the winding error. The first Blender run also could
not access the external OptiX cache inside the sandbox; rerun with cache access
succeeded. PowerShell treated redirected Blender deprecation warnings as a failed
command status; the final invocation explicitly returns Blender's actual exit
code (0), and the completion marker and all seven EXRs are checked.

Before the artist redirected the task, instrument-photo download attempts failed
with Wikimedia HTTP 429. No object model inference ran. A temporary OpenEXR package
was installed in the isolated UnReflect runtime and then removed, restoring its
package set. Other runtimes were not changed. The pre-existing round-17 files and
AGENTS edits remain untouched; the standard latest review log was updated by the
review encoder and is left outside this round's commit.

```powershell
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r20_eyes.py --shot music_videos/reporter_test/shots/reporter_a --prepare-only
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/audit_r20_eyes.py --shot music_videos/reporter_test/shots/reporter_a
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' -b --factory-startup --python scripts/render_r20_transport.py -- --shot music_videos/reporter_test/shots/reporter_a
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r20_eyes/review --fps 24 --out r20_eyes.mp4
& dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_video.py --shot music_videos/reporter_test/shots/reporter_a --name r20_source_fallback/review --fps shot --out r20_source_fallback.mp4
& dependencies/ellseg-runtime/.venv/Scripts/python.exe -B scripts/verify_r20_eyes.py --shot music_videos/reporter_test/shots/reporter_a
```

## 6. Starter prompt for round 21

> Read round20.md and docs/r20-eye-reconstruction.md. The artist explicitly says
> stay on eyes until fixed; do not resume instruments. Joint observation-based
> interior/reflection fitting now works in a restricted same-basis synthetic test,
> strongly dependent on supplied cornea and pupil assumptions. It is not a real
> reporter repair. Advance toward real corneal/pupil/occlusion geometry and joint
> observed-layer separation, preserving per-eye identity and exact fallback.
> Challenge the model on independently rendered textured eyes and uncertain geometry
> before treating near-zero algebraic residuals as reconstruction validation. Do not
> assume a photographed pupil is black or certify material from sum-fit alone.
> The full EyeNeRF neural method remains unrun; its current both-eyes-through-left
> texture path, rotational prior and minimum-shifted texture previews need explicit
> handling before use as evidence. Reuse the dual-ray optical transport and coupled
> replacement transaction; do not paint a flat catchlight disc or enable source
> removal without earned complete replacement support. Keep the real source where
> unreliable. Produce a labelled temporal review and round21.md. No generated
> performer texture, key retuning, special capture references or impossibility claims.
