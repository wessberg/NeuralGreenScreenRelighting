# Contributing

Read [architecture](docs/ARCHITECTURE.md) and [AGENTS.md](AGENTS.md) before changing a stage.
Keep the model engines in separate environments and document each stage's files and colour space.

Run `python -m pytest` for the contract tests. GPU checks need the environments in
[setup](docs/SETUP.md). For image changes, write to a fresh directory and compare the same
frames against the accepted result, including playback to check flicker.

Validate coordinate conventions in several directions when changing normals or rendering.
Run one GPU job at a time and leave Blender persistent data off. Keep source footage,
accepted keys and baselines unchanged, and use paths that work outside your checkout.

In a pull request, explain what changed in the output, how you checked it and what still
fails. Leave checkpoints, footage and generated caches out of the commit.
