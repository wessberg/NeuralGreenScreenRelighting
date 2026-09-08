# Round 3 — the ratio composite, and an external review that caught three live bugs

**Date:** 2026-09-08 · Follows `round2.md`, same working session

Round 2 ended with a proxy that cast real shadows but a delivery image that still came from a
gain map. This round built the path that actually reaches the plate, and then had it reviewed by
an outside model, which found more than it was comfortable to find.

---

## 1. The artist's verdict, which drove everything

He was shown a textured render of the proxy and named the failure frame by frame:

| frame | his words | what the light was doing |
|---|---|---|
| 70 | "looks OK" | heavily shadowed, grazing uplight |
| 296 | "90s CGI game-trailer" | backlit rim only |
| 186 | "plastic Barbie doll" | broad hard side light |
| 315 | "exactly like a character from The Sims" | broad, even, bright |

**His ranking maps exactly onto how much of her the light revealed.** That is the whole
diagnosis: rendering a reconstruction can only ever show the reconstruction, and a broad light
exposes precisely what a smooth shell does not have — pores, individual hair strands, fabric
weave, a corneal bulge. Frame 70 passed because 80 % of her was in darkness.

He also compared it to Apple Vision Pro Personas, which is the same architecture failing the same
way, and it is the sharpest description of the problem anyone has given.

---

## 2. The answer: relight the photograph, never render the person

`scripts/composite_ratio.py`

    out = plate × R + G,     R = E_target(rendered) / E_source(assumed)

Render the **irradiance field** on a neutral proxy, divide by the plate's own assumed lighting,
multiply the photograph. Every pore, hair and eye specular in the output is a photographed pixel;
the renderer contributes only the *change* in light — smooth almost everywhere and sharp exactly
where it should be, at a shadow edge. Those are the two things geometry is good at and a
per-pixel method cannot do at all.

The first frame out of this path was unmistakably a photograph rather than a character, and the
artist's assessment moved from "Sims" to "a big improvement… still a bit computer-graphics-ey".

**Two rules that are easy to get wrong, both now enforced in code:**

* The ratio pass must use the **plate's own lens and resolution**, or the multiplier does not line
  up with the photograph it multiplies. `stress_proxy.py --ratio` overrides the wider demo lens.
* **Clamp the top of the ratio only.** A shadow going dark is the point. `relight_sh.py` clamped
  both ends at 1.5 stops, capping the whole map at 2.83:1 — which is exactly why the gain map
  could never produce a terminator. Measured now: mean 4.6 stops of dynamic range, max 9.2.

---

## 3. Bugs the artist found by looking, and what they actually were

He caught every one of these from the image before any metric did.

**"Her hair is lit up SO much, with no falloff."** I diagnosed a specular lobe. **I was wrong.**
The measurement said the hot region was 41 531 pixels forming a disc (0.83 fill in its bbox; a
circle is 0.785) with coverage alpha 0.999 — solid geometry, not shading. It was the **emissive
demo bulb rendered into the irradiance pass**, 458× the median, which the composite then
multiplied into the plate. Mesh emitters *are* visible to camera rays, the exact opposite of
round 1's light-object trap. **Anything an irradiance pass can see ends up scaling the
photograph.**

**"Her jacket looks super shiny."** Specular does not scale with albedo, so a dark wool jacket was
receiving the same highlight as skin. Fixed with a per-material specular weight; wool is now 0.15
of skin's.

**"Vertical stripes… weird dark patches at the bottom… noisy holes in her arm that change
position every frame."** All one bug, and a real one: `make_proxy_mesh` back-projected X and Z
from the *original* depth and then changed that depth twice — the face-anchored relief rescale
(×0.455) and the normal-integrated detail. Move a vertex in depth without re-deriving X and Z and
it no longer projects to the pixel it came from.

| | before | after |
|---|---|---|
| vertical offset vs matte | −38.0 px | **−8.9 px** |
| matte left uncovered | 5.08 % | **0.62 %** |
| height ratio | 1.035 | **1.001** |

The 38 px shift left the bottom of the matte outside the proxy — the dark patches — and the
composite inpainted that gap from a different neighbourhood every frame, which is what flickered.

His instinct that "something disagrees about the size, aspect ratio, whatever" was right, and
comparing the rendered proxy silhouette against the matte is what turned it into a number. That
check is worth keeping as a permanent stage.

---

## 4. The external review — `codex review -c model="gpt-6-astra"`

Twelve findings, several with reproductions. It is the highest-value hour in this round.

**Three were live bugs:**

* the **skin mask was never normalised** — cast to float32 before the dtype test, so the /65535
  never fired: roughness reached **−32766** and specular weight **55704**. It was corrupting the
  render in flight;
* **proxy side walls were degenerate** — `[i, i+n, i+n, i]`, two distinct vertices, zero area.
  They blocked nothing, and I had written a comment claiming they would. The shell was open along
  its entire silhouette. Now 0 degenerate faces and 860 real wall quads;
* **inference crops clipped the performer** — when the 3:4 box exceeded frame height the width
  was shrunk, discarding **17–20 %** of matte pixels, which then received constant camera-facing
  normals. A fifth of her was being lit as a flat plane, and those normals feed the proxy *and*
  the relight. Now 0.00 % excluded, padded instead of shrunk.

**And it corrected a claim I had made to the artist.** `validate_proxy_shadows.py` did not test
self-shadowing: the review demonstrated that a shading-only hemisphere — `max(dot(n,l),0)`, no
shadow computation at all — passed all five of its checks. My "constraint 1 is green, the shadow
travels 14.5 % of face height" was **measuring shading**, the exact thing this project keeps
insisting is not shadowing.

Rewritten to measure occlusion **by difference**: each light is rendered twice, once with the
proxy's `visible_shadow` off, and the difference is the cast shadow and nothing else. A flat card
scores exactly zero. Real result: occlusion energy **10.9 %** and **13.1 %** of the lit image,
pattern IoU **0.359**, correlation **0.662**. The shadows are real; the old test was not what
proved it.

It also no longer asserts *which way* the shadow moves — a second overclaim. Under an uplight the
chin occludes the neck, an area far larger than the nose shadow on the brow, so the occlusion
centroid sits low under **both** lights (0.859 and 0.735). What separates occlusion from shading
is that the pattern depends on the light at all.

Other findings, all fixed: inverted cap winding (front faces pointed away from camera, leaving the
`RANDOM_WALK_SKIN` volume inside-out); `dominant_light` biased by the camera-visible hemisphere so
a flat ambient reported a frontal source at concentration 0.678 and the specular path invented a
highlight from nothing (now 0.0001); `video_depth.py` still writing a hardcoded 768×432 after its
consumers moved to a plate-derived resolution; `make_albedo` round-tripping the whole albedo
through uint8 and zeroing 8.9 % of positive foreground samples; `master_export` ignoring
`shot.conform` and turning a 150-frame 59.94 shot into 6.25 s of silent slow motion;
`neural_normals` accepting a convention report whose own verdict was FAIL; `--anchor-frame` parsed
and never used; producer and consumer disagreeing about the ratio-render directory.

---

## 5. GPU, because the artist saw his 5080 at 62 W

Cycles **is** on OPTIX. The card is starved, not unused. Render time barely depends on samples:

| samples | persistent off | persistent on |
|---|---|---|
| 4 | 2.21 s | 1.89 s |
| 128 | 4.06 s | 2.90 s |

**About 2 s of every frame is fixed Blender overhead** — scene sync, device upload, film init, EXR
write — and only a fraction is ray tracing. Confirmed from the other side: GPU 3.29 s vs CPU
5.01 s on the real scene is only 1.5×, because both pay the same fixed cost.

Consequences, which are the opposite of the intuition: **samples are nearly free, so spend them**;
moving more work to the GPU will not speed up rendering; and `use_persistent_data` is *not* the
way out — measured, it serves stale results and the error grows through a sequence (mean error
reaching 1.335 by the fourth frame).

`scripts/gpu_image.py` moves the per-pixel stages onto torch anyway (`cv2.cuda` reports 0 devices
in this wheel, so OpenCV cannot be accelerated at all). Its Gaussian initially differed from cv2
by 0.217 — the interior was exact and the error was entirely border convention, cv2 using
reflect101 rather than replicate. Now 2.4e-7.

---

## 6. Capture format, now fully specified — `capture_format.json`

**Panasonic LUMIX S5 IIX, 6K 3:2 open gate 5952×3968, HEVC 10-bit 4:2:0 Long GOP 200 Mbps,
V-Log / V-Gamut · SIRUI Saturn 35 mm anamorphic, 1.6× (CONFIRMED) · 180° shutter at 24 fps ·
WB 5600 K · delivering 3840×1600.**

5952/3968 = 1.5000 exactly; desqueezed by 1.6 that is 2.400:1; delivery is 2.40:1. Camera mode,
lens squeeze and delivery frame agree with no crop, and there is 2.48× linear oversampling in
hand.

**The recording mode is CHOSEN: `c58k_17_9`** — 5.8K 17:9, ProRes 422 HQ, all-intra, 4:2:2,
24p native, LPCM. Decided on keying grounds, accepting a 79.4 % width crop and 1.91× oversampling
in exchange for a codec that does not fight the matte. Open gate is Long GOP only on this body and
there is no external recorder, so ProRes RAW was never available. The trade, for the record:

| | 6K 3:2 open gate | 5.8K 17:9 |
|---|---|---|
| crop to reach 2.40:1 | none | 79.4 % of width |
| oversampling | 2.48× | 1.91× |
| hFOV at 35 mm | 78.9° | 64.7° |
| codec | 4:2:0 **Long GOP** | **ProRes 422 HQ, all-intra** |
| storage | 1.5 GB/min | 12 GB/min |

**Consequence for ingest:** all-intra means **no transcode is needed before keying**, which
removes one of the three blocking steps outright. 4:2:2 still needs chroma upsampled, and the
1.6× desqueeze still has to happen before any estimator. `capture_profile.py` derives all of this
from the profile, so a different camera later changes one field rather than any code.

`capture_format.json` was rebuilt after the artist caught it contradicting itself: a stale
vertical FOV of 27°, mode-independent FOV scalars that cannot exist, a recording mode asserted as
chosen while another key said it was open, and keying implications written for one mode presented
as universal. It now separates **stated facts** from **generated derivations**.

---

## 6b. `E_source` is now MEASURED, and the reference ball is retired for good

**Decision, and it is final: the artist will not shoot a chrome ball or a grey card**, and this is
intended to become a generic tool other artists can use on footage where no such reference exists
and never will. Every method that depends on an on-set procedure is out of scope. Do not propose
one again — it has been raised three times and answered.

So `E_source` is inferred from what every shot already has: the plate, the matte and the neural
normals. `scripts/estimate_source_light.py` fits SH-L2 lighting by robustly regressing skin
radiance against the measured normals. Lambertian irradiance is band-limited to SH-L2, so nine
coefficients per channel capture it; skin is used because its albedo is near-uniform within one
performer, and a Huber loss stops a forehead specular or a dark brow from dragging the fit.

**The albedo/lighting scale degeneracy is harmless here and should not be apologised for.** The
fit recovers `albedo × L`, not `L` — and `E_source` only ever appears as the denominator of a
ratio against a plate carrying the *same* albedo, so it cancels. What survives is relative albedo,
which is exactly what should survive.

`--selftest` recovers known lighting from a synthetic plate built on this shot's real normals,
through per-pixel albedo variation and 3 % noise. **4/4 PASS**, relative irradiance error 0.4–4.0 %.

### Two honest caveats, both measured rather than assumed

**The pass criterion is the irradiance field, not the light direction — and that is not a lowered
bar.** `composite_ratio`, `make_albedo` and `relight_sh` all evaluate `E_source(n)` at the visible
normals and divide by it. **None of them uses a light direction anywhere.** The field is what is
consumed, so the field is what is tested.

**The direction really is unreliable, and the report now says so loudly.** On this shot the
sampled normals span `ny` from −1.000 to −0.0599, and **0.00 %** face away from camera. The fit has
*no data at all* about light arriving from behind, so the direction it extrapolates there is an
artefact. It reported a source behind the performer at 133° from the old guess — that number is
meaningless and has been removed from the report rather than presented as a finding. It would have
been an easy and impressive-sounding thing to claim.

### What the measurement actually shows

Compared over the visible normals, both mean-normalised:

| measured vs the old hand-written guess | |
|---|---|
| relative RMS difference | **24.1 %** |
| correlation | 0.753 |
| spread p5–p95, measured | **0.973** |
| spread p5–p95, assumed | 0.432 |

**The real plate is about 2.3× more directionally lit than "ambient 0.55 + key 0.45" allowed.**
That is the substantive error in rounds 2 and 3: not a wrong direction, a wrong *amount of
directionality*, and it biased every ratio by roughly a quarter.

Wired into `composite_ratio.py` and `make_albedo.py`, which now print which model they used and
fall back to the guess only via an explicit `--assumed-source` flag kept for comparison. Albedo
has been regenerated for all 150 frames against the measured model. **The ratio composite has not
been re-run against it yet** — that is round 4's first measurement, and the current
`stress_ratio_composite.mp4` still uses the old assumption.

---

## 7. Honest status against the hard constraints

The table in `AGENTS.md` is the definition of done. Where it stands:

**Genuinely delivered this round:** the ratio composite path; real self-shadowing, now verified by
a test that a flat card fails; additive specular with per-material weight; proxy geometry that
projects correctly and is closed; V-Log decode with self-checks; plate-derived working resolution
throughout; capture profiles for both candidate modes.

**Edges, raised by the artist at the end of the round:** the silhouette reads as slightly
aliased or ragged. Measured before guessing, and the two obvious explanations are both ruled out:
the matte carries **12 294 partial-coverage pixels** with a reasonably flat distribution across
alpha, so it is genuinely antialiased; and the proxy covers **99.5 %** of the rim, with only 0.6 %
of rim pixels missing geometry. Two live hypotheses for round 4, in order:

1. **The side walls I added this round are the prime suspect.** Fixing the degenerate walls put
   860 real quads exactly at the silhouette, nearly edge-on to camera. Grazing-angle geometry
   renders dark and noisy, and multiplying the plate by a dark ragged rim would look exactly like
   this. Test by rendering the ratio pass with walls suppressed and comparing the rim.
2. **The artist's own hypothesis: the chroma key needs a little feather.** Cheap to test —
   the alpha distribution is mildly U-shaped, so a small feather before compositing may be all it
   needs.

**Still not done, and not pretended otherwise:** the composite runs against a black void rather
than an environment, so there is no bounce and no background integration. Eyes get no catchlight.
Hair shadows as a mass. `E_source` is still assumed. Rows 9–12 — MTF, grain, chroma, motion blur —
are untouched. The artist's own footage cannot be ingested at all yet: no desqueeze stage, no
all-intra transcode, no chroma upsample.

His current read: **"the render is much better now"**, and of the best frames earlier,
**"much better… but not quite outside the uncanny valley."** That is the correct assessment and it
should not be rounded up.

---

## 8. Decisions carried forward — do not re-litigate

Everything in round 1 §5 and round 2 §9 still stands. Added:

- **Never render the performer and show that.** The proxy exists to compute ratios and shadows.
  Any pipeline that puts rendered skin on screen has already failed, at any quality setting.
- **Anything an irradiance pass can see scales the photograph.** Emitters, props, the bulb —
  hide them from camera rays in any pass whose output is a multiplier.
- **Iterate at `--length 150`, not 360.** The full station tour still fits, at 2.4× the speed.
  The artist asked for this explicitly and it is the right default for a look.
- **Get the external review.** `codex review -c model="gpt-6-astra"` found three live bugs and one
  false claim in about an hour. Do it again before the next handoff.
- **No on-set reference, ever.** No chrome ball, no grey card, no colour chart as a *requirement*.
  This is a generic tool for other artists' footage as well, so anything that needs a procedure on
  the day is out of scope. `E_source` is inferred; if something else needs measuring, infer it.
- **Recording mode is `c58k_17_9`.** Build ingest for that first.

---

## 9. Open questions for the artist

**`E_source` is answered** (§6b) and **the recording mode is chosen** (§6). What remains needs
nothing on set beyond information the artist already has:

1. **His own interocular distance, measured once with a ruler.** Removes the 63 mm adult prior
   from every camera solve, permanently. This is the one remaining physical measurement, it is
   taken once rather than per take, and there is no way to infer it from footage.
2. **The T-stop he shoots at**, so anamorphic bokeh magnitude can be matched. The oval *shape* is
   already measured (`aperture_ratio` 1.6 → 1.6:1); the blur *amount* depends on the taking stop.
   If he would rather not record it, round 4 can fit it from a defocused background in his own
   footage — worth trying before asking.
3. **Whether V-Log constants can be verified without a chart.** `vlog.py --selftest` proves the
   transcription is self-consistent but cannot prove it matches this body. Given the no-reference
   rule, the fallback is to validate against the known black floor (V-Log 0 IRE sits at 0.125) and
   a skin-tone prior rather than a shot chart.

---

## 10. Starter prompt for round 4

> Read `rounds/round3.md` first, then `round2.md` and `round1.md` for context.
>
> The ratio composite works: `out = plate × R + G` relights the real photograph, keeps every pore
> and hair, and casts self-shadows verified by a test a flat card fails. The artist's verdict is
> "much better, but not quite outside the uncanny valley". Round 4 closes that gap and makes his
> own footage ingestable.
>
> **1. Eyes.** The strongest remaining uncanny-valley cue, and the cheapest to fix.
> `face_track.py` already runs mediapipe's 478-point iris-refined model and computes both iris
> centres (landmarks 468 and 473) — it currently uses them for IPD and throws the positions away.
> Store them, add two small spheres to the proxy at the tracked iris positions at face depth with
> a cornea-like smooth specular material, and let Cycles render a **real mirror reflection of the
> actual source**: correct position, correct colour, correctly occluded, moving on its own as she
> turns. No new estimator and almost no cost, because the proxy is already being rendered. Eyes
> with no catchlight are what make a Vision Pro Persona read as synthetic.
>
> **2. Put her in the environment.** The composite currently runs against a black void, so there
> is no bounce, no colour spill from the room, and no background integration at all. Render the
> ratio pass inside `scene_library/night_restaurant` instead of the test rig, and composite over
> the per-frame background. This is also what makes constraint 2 (received shadows) testable:
> put geometry between a practical and the actor and confirm the shadow lands on her and stops at
> her silhouette.
>
> **3. Ingest for `c58k_17_9`**, his chosen mode, which currently cannot be processed at all.
> Being all-intra, the transcode step is NOT needed — build `scripts/desqueeze.py` (1.6×, before
> any estimator; `anamorphic.py` has the maths and nothing calls it on a plate), chroma upsampling
> before keying, and the 79.4 % width crop to reach 2.40:1. **Autodetect everything detectable**,
> because this is meant to work on other artists' footage: ffprobe gives resolution, chroma, GOP
> structure and colour tags directly, and log-vs-linear can be inferred from the black floor
> sitting at V-Log's 0.125 rather than 0 when tags are absent. **Squeeze cannot be detected and
> must stay declared** — that is the one field a user has to supply.
>
> **4. Per-material properties from `facebook/sapiens2-seg-1b`**, replacing the red-dominance skin
> mask heuristic in `make_albedo.py`. Its 29-class taxonomy splits lips/teeth/tongue, which gives
> wet-surface shading for free and would let hair, wool and skin each take their own roughness and
> specular weight instead of a two-way lerp.
>
> **5. The ragged silhouette** (§7): test the side-wall hypothesis first by rendering the ratio
> pass with walls suppressed, then the artist's feather hypothesis. Measurements are in §7 so the
> ruled-out causes are not re-investigated.
>
> **6. Re-run the ratio composite against the MEASURED `E_source`** and compare. It is wired but
> the current video still uses the old assumption, which differs by 24 % relative RMS over the
> visible normals. Expect the biggest change on strongly shaded areas, because the real plate is
> about 2.3× more directionally lit than the guess allowed.
>
> Then the plate-matching group (rows 9–12): measured MTF blur, 4:2:0 chroma match, measured grain
> before quantisation, and 180° motion blur on the CG.
>
> **Iterate at `--length 150`** — the full eleven-station tour, 2.4× faster than 360. And get the
> external review before handing off: `codex review --base <round-start> -c model="gpt-6-astra"`
> found three live bugs and one false claim last time.
