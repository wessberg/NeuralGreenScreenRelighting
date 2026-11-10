# Neural green-screen relighting

Relight actors filmed against a green screen using neural surface normals and Blender
lights. [CorridorKey 2.0](https://github.com/nikopueringer/CorridorKey2.0) extracts the
foreground and alpha. [EyeClean](https://github.com/wessberg/eyeclean) removes the studio
reflections from the eyes, and Blender renders new ones from the scene lights.

The compositor changes the lighting on the photographed actor. A depth-and-normal proxy
supplies shadows and occlusion, so the output keeps the plate's skin texture, hair and clothing.

## Neural normals

[Sapiens2](https://huggingface.co/facebook/sapiens2-normal-1b) estimates surface directions
across the face and body. These normals let a moving light shade the nose, brow, cheeks
and clothing differently.

The estimator receives an sRGB crop of the keyed actor against neutral grey. We validate
the camera-to-Blender axis mapping before using the normals. They also supply fine relief
for the proxy mesh and help estimate the lighting in the original plate.

See [neural_normals.py](scripts/neural_normals.py),
[make_proxy_mesh.py](scripts/make_proxy_mesh.py) and
[light_geometry.py](scripts/light_geometry.py).

## Skin shading and specular

The proxy misses some facial relief. Meso shading uses the neural normals to shade those
smaller features under each scene light. The compositor measures where to blend between
the proxy and normal detail.

Skin specular uses GGX/Smith/Schlick shading with the light's direction, colour, size and
occlusion. With smooth neural normals, the lobe can make skin look like a shiny Barbie doll.
We broaden and reduce it, scale it against the photographed skin, and break it up with
texture from the plate.

The accepted reporter setting, `r44_m50_s05`, uses:

| Control | Setting |
|---|---|
| Meso strength | 0.5 |
| Skin specular gain | 0.5 |
| Skin roughness | 0.55 |
| Skin Fresnel F0 | 0.028 |
| Texture modulation power | 2 |

These are artistic compromises for this shot; the raw CLI defaults differ. The texture
factor stands in for pore detail that the normals cannot recover. The result improved
the skin but still falls short of the realism we're after.

[meso_specular.py](scripts/meso_specular.py) implements the shading.
The [integration guide](docs/INTEGRATION.md#accepted-skin-and-wet-eye-settings) has the full command.

## Eye reflections

EyeClean returns signed float corrections in the foreground's linear colour space.
The compositor applies them before relighting and checks the source hashes, so patches
from an old key cannot be applied after re-keying.

Blender renders the new reflections using a fitted cornea, wet sclera and socket.
The compositor adds them through the tracked eyelids. If EyeClean leaves an eye untouched,
we keep its original reflection and skip the new one. A separate pass removes studio
glints from the white of the eye.

## Pipeline

1. Decode the plate's colour and desqueeze it before estimating geometry.
2. Key with CorridorKey 2.0. Save straight foreground and coverage alpha as separate EXRs.
3. Estimate depth, face and eye tracks, and neural normals.
4. Run EyeClean on the keyed foreground.
5. Render proxy irradiance and eye reflections in Blender/Cycles.
6. Apply the eye corrections, lighting ratio, skin shading and new reflections in linear Rec.709.

## Use it

Follow [setup](docs/SETUP.md) to install the model environments and Blender.
The adapter package installs with:

```
python -m pip install -e .
ngsr --help
```

From another Python project:

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

The wheel contains the adapters and stage launcher. Geometry, rendering and compositing
still run from this checkout. See [integration](docs/INTEGRATION.md) for inputs and outputs,
and [architecture](docs/ARCHITECTURE.md) for the code layout.

## Reporter example

The [reporter example](examples/README.md) recomputes keying, EyeClean and normals, then
uses the existing Blender geometry and light passes. It writes linear RGBA frames,
labelled stills and comparison videos. Footage and weights are kept outside Git.

The v1 and v2 keys look very similar on this shot, including in an eight-frame v2 control
without hints. We have not demonstrated a visible keying improvement here.

Single-view geometry is approximate, normals can flicker, and the captured hair and lip
highlights remain. Hair highlight removal was rejected and is off. Camera-log ingest is
incomplete; V-Log/V-Gamut footage needs decoding upstream.

## Licence

The original code is MIT. Dependencies and weights have separate terms, including
CorridorKey's custom licence and the Sapiens2 model licence.
Read [third-party notices](THIRD_PARTY_NOTICES.md) before distributing a configured pipeline.

[Contributing](CONTRIBUTING.md) | [Release checklist](docs/RELEASE.md) |
[Development history](docs/history/README.md)
