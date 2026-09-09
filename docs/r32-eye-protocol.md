# Round 32 protocol — same-frame anatomical completion of the corneal reflection

Frozen before any withheld-pixel error was computed. Development of the detector used
source frames 700–707, which are therefore seen; the evaluation below scores pixels
that no part of the method has read, and its design, sites, controls and reported
statistics are fixed here in advance.

## 1. Why this is not one of the previous thirty rounds

| Rounds | What was tried | Outcome |
|---|---|---|
| 8–12 | recover the hidden iris from other frames (centre/scale, affine, torsion) | zero confident donors under the reflection |
| 13–19 | EllSeg, RITnet, RAFT, CoTracker3, UnReflectAnything | proposals and tracks only; no donor coverage, no repair |
| 20–24 | forward Cycles eye optics, inverse layer fits on synthetic fixtures | fits worse than source on iris; pupils dark, patchy, negative |
| 25–26 | real source arcs, inverse ring geometry, joint families | geometry ambiguous across ±40° yaw |
| 27–30 | joint appearance + spherical-Gaussian light inverse on real eyes | catchlight error 25–77 display levels; "spatial backing" paints the spot into the pupil |
| 31 | released DiffusionRenderer, generative | rejected twice by the artist |

Every one of those either **searched elsewhere** for the occluded iris or **fitted and
subtracted a physical layer**. This round does neither. It completes the footprint from
the same frame's own eye, which is what round 31's starter prompt asked for.

## 2. Measured facts this round starts from

Measured on the shot before the method was designed; all are re-reported by the run.

1. **The reflection is unclipped.** Peak display luminance inside the limbus is
   0.7843–0.8823 across source 700–723, with **zero pixels at 255**. The corrupted
   region is a sum, not a sensor hole.
2. **The footprint is bright core plus dark rim.** The plate is tagged `bt709` limited
   range and was expanded on ingest (`logs/extract_command.json`), so the undershoot
   beside the highlight reaches exactly 0 in green and blue. Removal that ignores the
   rim leaves a black outline.
3. **The pupil edge is not resolvable in this plate.** The radial luminance profile
   rises smoothly from ≈0.01 to ≈0.13 linear with no step, and the steepest-gradient
   radius moves from 2.5 to 6.5 native pixels when the assumed centre moves by two.
   At a 24-pixel iris in 8-bit 4:2:0 H.264, a dark brown pupil inside a dark brown
   iris is below what the plate carries. **This round therefore does not claim to
   separate pupil from iris.** It anchors on the limbus, which is well defined, and
   lets the measured radial profile carry whatever darkening the plate does resolve.

## 3. Method, fixed

Per eye, per frame, entirely inside a local crop:

1. **Limbus** from the four mediapipe cardinal iris landmarks (axis-aligned ellipse).
   Checked against the plate at 24× magnification before being trusted. EllSeg's
   ellipse is not used: its softargmax pupil centres on the highlight itself.
2. **Visibility** from the eyelid polygon, supersampled ×4, requiring >0.85 native
   pixels of clearance inside the lid.
3. **Appearance model** = per-channel median radial profile in bins of 0.055 normalized
   radius out to 1.18, multiplied by a smooth angular shading gain (constant plus the
   first two harmonics, ridge 0.05 toward unity). The angular term exists because the
   lash line and lid shadow darken the top of the iris and are not radially symmetric.
4. **Footprint** = hysteresis on the absolute residual against that model. The spread
   is a MAD over all visible pixels, **frozen at the first iteration**; high level
   `max(4.0σ, 0.012)` seeds, low level `max(1.5σ, 0.004)` grows, connectivity is
   8-way, the component must contain a seed, everything is confined within 8.0 native
   pixels of the brightest unexplained pixel, and the result is dilated by one pixel.
5. **Completion** = the model plus an angular interpolation of the *residual*, taken
   from the same eye, same frame, same normalized radius (±0.075), from up to three
   nearest unmasked donors on each side, inverse-angular-distance weighted, with a
   two-pixel collar excluded from the donor set. Filling the residual rather than the
   value is what preserves whatever radial step the plate does contain.
6. **Abstention.** Any eye whose footprint exceeds 25 % of its visible area, whose
   donor annulus is under 60 pixels, whose visible area is under 90 pixels, or whose
   track is missing, keeps **bit-exact source**.

No generated material. No cross-frame correspondence. No layer subtraction. No scene
reflection is added, and `world_registration_validated` stays false.

## 4. Withheld-pixel evaluation — frozen design

The method is scored on pixels it never sees, in the same eyes, on real iris texture.

* **Sites.** For each repaired eye instance, the actual footprint mask is rotated about
  the limbus centre by 60°, 120°, 180°, 240° and 300°. A rotated copy is **rejected**
  if it overlaps the real footprint or its two-pixel collar, if any of its pixels fall
  outside the visible eye, or if it leaves the limbus. Accepted copies are the test
  sites. Rotation preserves footprint size, shape and normalized radius exactly, so
  each site poses the same completion problem at a different angle.
* **Procedure.** The site is added to the mask, the model is refitted with both the
  real footprint and the site excluded, and the identical completion runs. The
  prediction is compared with the true observed pixels at the site.
* **Statistics reported.** Median and p95 absolute display RGB error (0–255 levels)
  over all site pixels, split by whether a pixel lies inside or outside the profile's
  steepest-step radius, and reported per eye.
* **Controls, all on identical pixels.**
  1. `profile` — fill with the appearance model alone, no angular residual. Isolates
     what the angular interpolation adds.
  2. `annulus_median` — fill with the median of the same annulus. A naive baseline.
  3. `single_gain` — fill the whole site with one flat linear RGB, fitted on the
     surrounding ring. This is round 31's failure mode reproduced on the same pixels,
     so the comparison is paired rather than across rounds.
  4. `texture_floor` — the median absolute deviation of real unmasked pixels from the
     same model at the same radii. This is the plate's own angular texture variation:
     no completion can beat it, and a completion near it is as good as an observed
     pixel at that radius.

**No threshold in this document is a pass/fail gate on the repair.** The artist's eye
is the acceptance bar, and the review video is the artefact he judges. These numbers
exist so a claim about the fill can be checked, and so the `single_gain` control shows
whether the change from round 31 is real.

## 5. Temporal reporting, fixed

Round 31's per-frame ring-fitted gain swung red 1.73× between source 701 and 722, with
R/B warmth 2.162 against 1.307, and that is what the artist saw as one frame reading
wrong and another acceptable. The same statistic is reported here: per eye, across all
repaired frames, the min/max ratio per channel of the mean linear fill, and the R/B
ratio at source 701 and 722. Nothing in this round fits a per-frame gain, so the
expectation is that the fill tracks the eye's own illumination rather than a solver.

## 6. Claims this round may not make

* Not a pupil/iris separation — see §2.3.
* Not clean recovered material. The fill is a plausible completion from surrounding
  anatomy, and the true radiance under the reflection remains unobserved.
* Not a scene-light replacement. Removal leaves an eye with no catchlight, which the
  artist must judge; the replacement half of the task is unbuilt and still requires
  validated world registration.
* Not a production default, and not artist acceptance.
* Not a universal result: this is one plate, at one resolution, with one reflection.

---

## 7. Amendments made after this freeze — READ WITH §2 AND §3

This document is retained unchanged as the pre-scoring record. The shipped method
then diverged from it in three ways, all after development-frame inspection and all
**before** any withheld-pixel score was computed. `rounds/round32.md` §4 is the
authority.

1. **§3.3 model region.** Frozen text builds the profile out to ρ = 1.18. Shipped
   stops at ρ = 0.88. Reaching past the limbus put the sclera inside the model, and
   the sclera is bright and sits left and right of the iris but not above and below
   it, so it is not radially symmetric about the iris centre. Residuals were then
   large everywhere and 106–197 of about 460 visible pixels tripped the 4σ level
   while the highlight itself did not stand out.
2. **§3.4 footprint.** Frozen text takes symmetric hysteresis on the absolute
   residual. Shipped grows on the residual's **sign** — positive core, then the
   negative rim touching it — and then **does not repair the rim at all**. The rim is
   coding undershoot clipped at legal black, but it is also the only dark material
   the plate carries around the pupil, and the artist rejected replacing it.
   The frozen ratio-based size guard was also replaced by an absolute one.
3. **§2.3 reasoning.** The claim that no pupil edge is resolvable was an artefact of
   profiling about the limbus centre and is corrected in `rounds/round32.md` §3.6.
   The limbus anchor itself is unchanged, for a different and narrower reason.

**Outcome:** the result was rejected by the artist for colour flicker inside the
pupil. The withheld-pixel design in §4 ran exactly as frozen, but its rotation scheme
rejects 1,091 of 1,400 sites for self-overlap — at ρ ≈ 0.3–0.6 a footprint's angular
width leaves little room for a disjoint copy. That is a design weakness to fix before
reusing it, not a result.
