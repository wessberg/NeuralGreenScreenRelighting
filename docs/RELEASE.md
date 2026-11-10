# Release preparation

The code is prepared for review; this document does not authorize publishing or uploading media.

- [x] Contract suite: 10 CPU/CUDA tests pass, including external-pass HDR/alpha and patch preservation.
- [x] Wheel and source archive build; the wheel CLI starts in a clean environment without GPU engines.
- [ ] Make the EyeClean r57 bundle available to new users: its pinned registry currently has no download URL.
- [ ] Before PyPI publication, replace the direct Git client dependency with a published version (Git/checkout installs work).
- [x] Complete the 150-frame reporter GPU smoke run; record exact dependencies, hashes and reused passes.
      All linear RGBA frames are finite, retain key alpha, preserve EyeClean edits during sclera
      cleanup, and add no scene reflections to unrepaired eyes. New artistic acceptance is pending.
- [ ] Inspect every changed eye against source/accepted results and review the comparison video.
- [ ] Obtain artistic acceptance of the new EyeClean/keyer combination. The round-44 approval
      applies to the old baseline, not automatically to updated dependencies.
- [ ] Review third-party licences for the intended distribution and use.
- [ ] Confirm rights to every image/video included in a public example. The reporter source
      is stock media; local processing permission does not by itself authorize redistribution.
- [ ] Scan the tracked tree and full Git history for secrets, private media and machine paths.
      Ignoring a file does not remove it from existing history.
- [ ] Prefer a clean source export or new public repository over publishing the development
      history, which contains local production metadata. No history rewrite has been performed.
- [x] Inspect built artifacts: no dependencies, model weights, reporter footage or generated caches bundled.
- [x] Document platform coverage: Windows/NVIDIA tested; Linux and other GPUs require validation.

Automated common-credential/private-key pattern scans found no matches in the tracked tree or
93 reachable commits. This does not replace the broader media, machine-path and licence review.

The adapter wheel intentionally contains only `ngsr`. The source archive excludes development
fixtures and historical logs via `.gitattributes`; the checkout retains them for reproducibility.
Generated reporter media remains local until its redistribution rights are confirmed.
