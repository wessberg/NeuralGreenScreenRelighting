# Release checklist

## Checked

- [x] Ten CPU/CUDA tests pass, including HDR/alpha output and float patch preservation.
- [x] Wheel and source archive build. The wheel CLI starts in a clean environment.
- [x] The 150-frame reporter run completes with dependency hashes and reused passes recorded.
      Linear RGBA output is finite and retains the key's alpha. Sclera cleanup preserves
      EyeClean's edits; new reflections are skipped on unrepaired eyes.
- [x] Built artifacts exclude dependencies, weights, reporter footage and generated caches.
- [x] Platform coverage is documented: Windows/NVIDIA tested; Linux and other GPUs untested.

The credential-pattern scan found no matches in the tracked tree or 93 reachable commits.
Private media and local paths still need review.

## Before release

- [ ] Provide a download for EyeClean r57; its pinned registry has no URL.
- [ ] Replace the Git client dependency with a published version before publishing to PyPI.
      Git and checkout installs work.
- [ ] Review every changed eye and the comparison video, then get artistic acceptance.
      The old R44 approval doesn't cover the updated models.
- [ ] Check dependency licences for the intended distribution and use.
- [ ] Check the stock-footage licence before publishing reporter stills or clips.
- [ ] Review the tracked tree and history for private media, local paths and secrets.
      Ignoring a file leaves it in history.
- [ ] Choose how to publish the source. The development history contains local production
      metadata; it has not been rewritten.

The wheel contains `ngsr`. Both the source package and Git export exclude development
fixtures and logs. Reporter renders stay local until their redistribution rights are confirmed.
