# Neural green-screen relighting

Relight a photographed actor filmed against a green screen. **Neural surface normals drive
the actor's shading, a dedicated skin shading/specular stage controls the skin response,
and Blender supplies light transport, shadows and eye reflections.**
[CorridorKey 2.0](https://github.com/nikopueringer/CorridorKey2.0) extracts the actor;
[EyeClean](https://github.com/wessberg/eyeclean) removes the original eye reflections before
the new scene's reflections are added.

The performer remains a photograph. A proxy supplies lighting and occlusion; it is never
the visible actor. This is a research pipeline being prepared for release, with known
quality limits, rather than a finished photorealistic reconstruction system.

## Neural surface normals

[Sapiens2](https://huggingface.co/facebook/sapiens2-normal-1b) estimates the orientation of
the performer's surface: the nose, brow, cheeks, lips, clothing and other visible form.
Those neural normal vectors make illumination respond to the direction of the Blender
lights, including changes across a face when a light is close to it.

The estimator sees the keyed actor cropped to its matte and placed against neutral grey,
with the expected sRGB input. Its camera-to-Blender axis mapping is independently validated
and recorded. The resulting unit normals drive source-light estimation, fine proxy relief,
per-light surface shading and skin specular.

Implementation: [neural_normals.py](scripts/neural_normals.py),
[make_proxy_mesh.py](scripts/make_proxy_mesh.py) and
[light_geometry.py](scripts/light_geometry.py).

## Skin shading and the specular fix

A coarse proxy cannot reproduce all of a face's relief. Its lighting ratio alone can leave
the skin looking smooth or like a game character. The skin stage addresses that in two parts:

- **Normal-driven surface relief:** per-light meso shading restores the smaller variations
  described by the neural normals, while the Blender proxy supplies shadows, falloff and
  larger-scale form. The transition scale is measured from the proxy and normal fields.
- **Restrained, textured specular:** a GGX/Smith/Schlick skin lobe follows the actual scene
  light's direction, level, colour, angular size and occlusion. Its strength is scaled against
  the photographed skin, broadened and broken up with the plate's own texture. This avoids
  the smooth shiny band that previously gave the face a Barbie-like appearance.

The artist-accepted **R44 development preset** is:

| Control | Setting |
|---|---|
| Meso shading strength | 0.5 |
| Skin specular gain | 0.5 |
| Skin roughness | 0.55 |
| Skin Fresnel F0 | 0.028 |
| Photographic texture modulation | power 2 |

These are explicit preset settings, not universal material measurements or the raw CLI
defaults. Neural normals do not recover pore-scale geometry; texture modulation is a
heuristic. The accepted result improved the skin, but did not meet the full realism goal.

Implementation: [meso_specular.py](scripts/meso_specular.py) and
[composite_ratio.py](scripts/composite_ratio.py).
The [integration guide](docs/INTEGRATION.md#accepted-skin--wet-eye-development-preset)
contains the complete command.

## EyeClean and Blender eye reflections

EyeClean runs in its own environment and returns signed float corrections in the keyed
foreground's linear colour space. New reflections are rendered from Blender lights using
the fitted cornea, wet sclera and socket geometry, then added through the tracked eyelids.
An EyeClean-abstained eye retains its original reflection and receives no added one.
The accepted scleral-glint cleanup is retained as a separate correction pass.

## Pipeline

1. Decode colour and desqueeze the plate before geometry estimation.
2. Key with CorridorKey 2.0 into separate straight foreground and coverage alpha EXRs.
   Preserve photographed pixels in the opaque core and apply the shot's despill settings.
3. Estimate temporal depth, face/eye geometry and Sapiens2 surface normals.
4. Run EyeClean in its own process on the keyed foreground. Keep its signed float corrections.
5. Render neutral proxy irradiance and wet-eye reflections in Blender/Cycles.
6. Apply EyeClean corrections, multiply by target/source irradiance, add restrained skin
   specular and the new eye reflections, then composite and grade. The optional accepted
   sclera cleanup is a separate float correction pass.

Corrections stay scene-linear Rec.709 until review/export. An EyeClean-abstained eye retains
its original reflection and receives no new one. Source hashes prevent applying corrections
made before a re-key.

## Get started

See [setup](docs/SETUP.md) for the three isolated environments, checkpoints and Blender.
The lightweight adapter package installs separately from the GPU engines:

```
python -m pip install -e .
ngsr --help
```

Call it from another codebase:

```python
from ngsr import run_eyeclean, run_stage

eyes = run_eyeclean(
    "/show/shot/key/fg", "/show/shot/composite/eyeclean",
    fps="24000/1001",
    engine=["/tools/eyeclean/.venv/bin/python", "-m", "eyeclean"],
)
run_stage(
    "composite", "/show/shot",
    repository="/tools/neural-green-screen-relighting",
    python="/tools/neural-green-screen-relighting/.venv/bin/python",
    options=["--render-dir", "/show/passes/irradiance", "--eye-patches", eyes,
             "--name", "relit", "--specular", "0", "--linear-output"],
)
```

[Integration](docs/INTEGRATION.md) describes stage contracts and the Blender-tour preset.
Geometry/render/composite stages currently require this source checkout; installing the
wheel alone supplies the engine adapters and stage launcher.

## Examples and validation

The [reporter example](examples/README.md) runs a fresh key, EyeClean and neural normals
against the existing development shot and Blender passes. Media and model weights stay
outside Git. It produces linear frames, labelled comparisons, close eye crops and a review
video. Cached Blender geometry/passes are explicitly identified in its report.
The reporter test demonstrates pipeline operation; it has not demonstrated a visible
keying-quality improvement over CorridorKey 1, including an eight-frame native no-hint control.

Remaining limits include approximate single-view geometry, temporal noise in neural normals,
capture-room hair/lip highlights, matte quality and incomplete camera-log ingest.
Hair highlight removal was rejected and remains off. Still frames and numerical checks
do not establish temporal or artistic acceptance.

## Documentation

- [Setup](docs/SETUP.md) ? [Integration](docs/INTEGRATION.md) ? [Architecture](docs/ARCHITECTURE.md)
- [Release checklist](docs/RELEASE.md) ? [Contributing](CONTRIBUTING.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md) ? [Development history](docs/history/README.md)

## Licence

This repository's original code is MIT. Dependencies and weights have separate terms.
In particular, CorridorKey uses a custom licence and Sapiens2 uses its own model licence.
They are downloaded separately and are not included in the adapter package.
See [third-party notices](THIRD_PARTY_NOTICES.md) before distributing a configured pipeline.
