# Contributing

Start with README.md, docs/ARCHITECTURE.md and AGENTS.md. Keep new integrations behind explicit
file contracts and leave heavyweight engines in isolated environments.

Run `python -m pytest` for CPU contract tests. GPU integration needs the setup in docs/SETUP.md.
Use a fresh output directory and compare against the accepted result at identical source frames.
A numerical score cannot substitute for inspecting images and temporal behaviour.

When changing normals or rendering, validate coordinate conventions in multiple directions.
Run one GPU workload at a time. Do not enable Blender persistent data. Do not overwrite source
footage, accepted keys or baselines. Avoid introducing dependencies on a developer's absolute paths.

Pull requests should describe the visible/behavioural change, validation, and remaining limits.
Do not include checkpoints, stock footage, private recordings or generated caches.
