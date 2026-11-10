# Third-party notices

The MIT licence at the repository root covers this project's original code. It does not
relicense dependencies, model weights, stock footage or Blender/FFmpeg distributions.
The package does not bundle those assets.

| Component | Upstream terms / use |
|---|---|
| [EyeClean](https://github.com/wessberg/eyeclean/blob/a3efafd6d2794cace9647251791f411bbeaf626f/LICENSE) | MIT code and released r57 weights. Its Blender training tools have different terms and are not used here. Detector weights retain their upstream terms. |
| [CorridorKey 2.0](https://github.com/nikopueringer/CorridorKey2.0/blob/485f5e9ab759ead6138cc990776184b59d6ea971/LICENSE) | Custom Corridor Key Licence v2.0 based on CC BY-NC-SA with additional permissions/restrictions. Commercial output is permitted; redistribution, resale, paid inference services and integration have restrictions. Organisations above the stated revenue threshold require plugin seats. Read the complete upstream licence. |
| [CorridorKey foundations](https://github.com/nikopueringer/CorridorKey2.0/blob/485f5e9ab759ead6138cc990776184b59d6ea971/THIRD_PARTY_NOTICES.md) | C-RADIO: NVIDIA Open Model Licence; MoGe: MIT with DINOv2 notices; RVM: Apache-2.0 code / CC-BY-4.0 weights. Downloaded separately. |
| [Sapiens2 normals](https://huggingface.co/facebook/sapiens2-normal-1b) | Sapiens2 model licence; consult the linked licence for allowed use. Not MIT merely because the Transformers integration is open source. |
| [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) | Check the selected checkpoint's terms; the existing pipeline uses Small. |
| [Blender](https://www.blender.org/about/license/) | Separate Blender distribution, invoked as a process. |
| [FFmpeg](https://ffmpeg.org/legal.html) | Licence depends on the selected build. No FFmpeg binary is redistributed here. |

Historical experiments reference NVIDIA DiffusionRenderer and other non-commercial models.
They are not required by the current key / EyeClean / normals / skin-specular composition path.
Retaining experimental code does not grant permission to redistribute those models.

Development reporter footage is identified in the shot metadata. It is not bundled in the
adapter package or source release export. Review the original provider's media licence before
publishing stills or clips.
