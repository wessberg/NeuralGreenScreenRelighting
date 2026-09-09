# Round 26: shared source geometry and withheld observed radiance

Frozen before inspecting the new source-805 eye boards or fitting this round.
Stay on eyes. This diagnostic cannot enable production removal/replacement.

Use round-25 source 730/760/790/820 as development observations, separately per
eye. Add source 805 (local 105) as a new source-only annotation check on reused
footage. Inspect only its native RGB before labelling nine visible limbus points,
four readable pupil points, the observed lid opening and incomplete reflection
exclusions. Use 2px uncertainty. These are assistant labels, not human truth.
Freeze source archive, native RGB/origin, labels, prior fit and protocol hashes.

Reuse round-25 train-only ring candidates. Fit the new target ring from the first
and last thirds of its visible arc; reserve its middle third. Donor pupil labels
are development conditioning. Only the first two target pupil points condition
geometry; reserve its last two. Rank by conditioning residuals only. No RGB
enters geometry, correspondence selection, support or ranking.

Jointly constrain five observations of each eye with shared corneal radius
7.2/7.8/8.4mm, iris depth 3.3/3.6/3.9mm and pupil decentration xy from
{-0.6,0,0.6}mm squared. Per-frame gaze/centre/depth use the ring families; pupil
radius may vary over 1.2..3.0mm in 0.3mm steps. These finite priors do not establish
anatomy. Keep the full factored family of compatible per-frame assignments at
each shared parameter tuple, not just the best sequence. Compare centred-pupil
and baseline-depth restrictions; report 1/2/3px pupil-label sensitivity from
the same residuals. Do not select thresholds using check outcomes.

Use the best conditioning-only joint hypothesis to transport each eligible native
target iris pixel to each same-eye donor through the refracted iris plane. This
assumes a rigid iris and zero relative torsion, and transports observed radiance,
not albedo. Numerically invert the donor ray map and check convergence. Require
all four bilinear donor pixels and their 3x3 BOX samples to remain on the candidate
iris and conservatively inside observed lids/outside reflection polygons. Use
2px plus sqrt(2) raster/BOX margin; also report 1px and 3px lid uncertainty.
Target RGB is withheld in its entirety until predictions/eligibility are stored.
This is a fresh computational RGB check on reused footage, not unseen-image
acceptance. There is no RGB fit or search, and donor frames never include target.

Compare a frontal ring-based centre/scale control on identical paired support.
Report each donor and their equal-weight linear-RGB mean, display error and
abstentions. Require at least three donors for the mean comparison. Inspect
coordinate variation across the best assignment at every compatible shared
tuple, and relative torsion -10/0/+10 degrees. These witnesses give a LOWER bound
on sampled ambiguity: witness agreement never certifies the full factored family.
No photometric cutoff promotes pixels into a clean or reliable mask.

The model keeps round-24's sharp limbus and single effective interface. Varying
iris depth is only a sensitivity test, not posterior-cornea/lens modelling.
3D lid/light occlusion, pupil deformation, real camera PSF, smooth limbus,
posterior optics, source-light decomposition and production scene registration
remain unresolved. No anatomical or clean-material claim is permitted.

Verify signed/asymmetric-camera inverse transport, centred compatibility,
decentration sign, empty-family/closure/complete-footprint rejection, target-RGB
substitution invariance, held-label isolation, family replay and source hashes.
Verify 300 exact keyed-eye fallbacks and protected production hashes. Produce
150 continuous source-left diagnostic-right frames at shot fps, plus labelled
held detail if needed; decode and inspect the review. End with round26.md.
