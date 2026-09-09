# Round 31: selective DiffusionRenderer trial

**Outcome: both the whole-eye replacement and the reflection-spot follow-up
were explicitly rejected by the artist.** Whole eyes become flat/golden with
distorted pupils; restricting the same generated albedo to reflection spots
leaves a light brown spot inside the pupil. This is not an acceptable repair.
See `rounds/round31.md` and `logs/r31_artist_review.json` under the shot.

The artist explicitly authorizes selective generation after round 30 and prefers
DiffusionRenderer. This supersedes older bans on generated eye detail. The current
custom optical solver is not close to a working solution; do not continue its
parameter search in this round.

Produce a short source/result video using released DiffusionRenderer, with both
eyes and a blink. Use the source face as input context. Estimate the required
G-buffers, then render under a controlled environment and the shot environment.
Composite only the visible inner eye region; preserve source outside that mask.
Report whether reflections actually change, whether gaze/iris colour/blink timing
survive, and whether seams/flicker make the result unusable. Generated albedo is
not an observed clean layer; appearance and temporal quality are the criteria.

The failure occurs before compositing: the inverse pass already invents different
iris/pupil appearance. Forward relighting and local colour matching cannot make
that invented content correct. A smaller mask preserves more original image but
does not fix the filled pixels. The next candidate must explicitly complete missing
pupil/iris content inside a mask, using surrounding anatomy as constraints. Do not
reuse this albedo as a clean fill, paint pupils brown/black, or claim that a newer
model is a proven solution before rendering and inspecting its result.

Start with the SVD release already checked out at
`dependencies/DiffusionRenderer` (8fcf0057ad3422139cd53281037025ff725d34e9).
Its runtime is isolated at `dependencies/diffusion-runtime/.venv`. Use current
Blackwell-compatible torch/CUDA, author diffusers/transformers versions and
explicit weight hashes. The original 24-frame/512-pixel configuration is documented
above 22 GB VRAM. Test smaller spatial dimensions, VAE slicing, CPU offloading and
forward chunking on the 16 GB RTX 5080, reporting the actual settings and costs.
Do not silently present this SVD trial as the newer Cosmos model.

The SVD code/weights license restricts use to research/evaluation; this round is
an evaluation, not adoption for commercial delivery. Cosmos uses a different
release and dependency stack; its quality and deployment remain separate questions.

Preserve source rate and colour metadata. Feed display-sRGB into the learned model,
and return decoded generated RGB to linear for masked blending. Use a fixed crop
derived from shot landmarks, preserving pixel aspect; an enlarged crop adds no
source detail. Masks derived from landmarks are proposals, with source fallback
for closures/invalid tracks. Keep full generated crops visible in a diagnostic
panel so a narrow blend cannot hide changes to gaze or anatomy.

Record actual outcomes, including failures. Review videos are required; another
analytic benchmark is not a substitute for attempting inference on the shot.

Author sources checked 2026-09-09:
[SVD code and runtime](https://github.com/nv-tlabs/diffusion-renderer),
[Cosmos release](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer),
[project and demonstrations](https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/).
