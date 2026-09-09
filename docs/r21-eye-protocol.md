# Round 21 frozen protocol: independent textured transport and geometry uncertainty

Frozen before rendering or fitting, 2026-09-09. Stay on eyes. This experiment is
nonperformer validation, not permission to change any real keyed eye.

24 square-pixel 64x64 frames; fx=fy=3000 px, centre=(31.5,31.5), apex at
(0,0,1300) mm. Anterior conic R=7.8 mm, eccentricity=.5, n=1.376, sag cap=2.18;
emissive iris plane at 3 mm, radius=5.85, nonblack pupil radius=1.8 mm.
Yaw=12*sin(2*pi*i/24), pitch=8*cos(2*pi*i/24) degrees. Independent elliptical
lid half-height=4*(1-exp(-((i-11)/1.4)^2)), half-width=8, world z=1299 mm.
These are declared fixture assumptions, not anatomical calibration.

Cycles independently tessellates/refraction-renders the eye using 512x512
procedural nonperformer textures, including fine angular iris structure, asymmetric
colour marks, a nonzero textured pupil and sharp environment rectangles. It never
uses the fitter basis or analytic trace to generate RGB. Source, clean transmitted
interior (black environment), and new-environment target are separate renders.
4096 samples, OPTIX, fixed seed 21, denoising and persistent data off, .01 pixel
filter. No neural RGB decoder or performer pixels are used. Geometry conventions
and optical simplifications remain shared with round 20: this is independent
rendering/basis evidence, not independent anatomy, scattering or real footage.

Whole-frame splits: selection 3,9,15,21; final test 5,11,17,23 (11 is closed);
remaining 16 are training. No selection or test RGB enters a layer fit. Selection
RGB chooses one finite geometry candidate by source-sum display MAE. Final test
RGB and all clean/target RGB are unavailable to fitting and selection. Use a
common mask across candidates: true visible interior eroded one pixel, minus
the .7 mm pupil boundary band. This ORACLE evaluation region is fixed before RGB
and shared by every arm; report its limitation, full visible-region supplements,
missing predicted visibility, and iris/pupil separately. No candidate-specific
score exclusions. Training uses that same fixed geometry-only region.

Compare a 12x12 continuous interior grid with a 12x12 iris plus separate 12x12
pupil grid, both fitting nonnegative unknown pupil RGB (never an oracle black
pupil). Pupil coordinates rescale to its own disc. Environment is 12x12; bounds
interior/pupil [0,1], environment [0,8]. No spatial or rotational regularizer.
Use bounded sparse least squares, tol=1e-9, LSMR tol=1e-10, max_iter=200;
report any unconverged channels without retuning. Zero-support columns are zero
and flagged unsupported, never called observed texture.

Split-pupil candidates: supplied true geometry, yaw bias +/-2 degrees, corneal
radius +/-5%, pupil radius +/-0.3 mm, lid height +/-0.4 mm (clamped at zero).
These nine one-at-a-time hypotheses are a sensitivity/finite selection test,
not joint continuous calibration. Keep the closed frame closed in every candidate.
Continuous-grid arm uses true geometry and is excluded from geometry selection.

Report source-sum, recovered transmitted layer, old reflection, and coupled
target errors separately (linear and display 0..255), against untouched source
and add-only controls. The target reflection is ORACLE independently rendered
target minus clean, so this tests separation conditional on known replacement
transport, not inferred scene lighting. No sum-fit score certifies material.
Compare solver results only on fixed identical masks; report sample/parameter
counts and all candidates, not just the best one. Preserve raw negative Monte
Carlo subtraction residuals in scoring; clamp only the displayed target-reflection
payload used to exercise the disabled transaction, recording any negatives.

Verify saved-fit scalar sampling, bounded values, whole-heldout-RGB substitution
input invariance, final-test substitution selection invariance, clean/target
substitution isolation, exact disabled coupled transactions, 300 keyed fallbacks
and 304 protected hashes. Hash protocol, code, rendered source and reference files.
Produce a chronological labelled 24-fps review (24 frames repeated four times),
decode every frame, inspect ordered contact sheets and native boards. Reuse the
existing real exact-source review by reference; do not manufacture real improvement.
Finish round21.md with findings, limitations and a round-22 starter prompt.
