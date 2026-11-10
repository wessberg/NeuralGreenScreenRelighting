# Third-party notices

The root MIT licence covers this project's original code. Dependencies, weights, footage,
Blender and FFmpeg retain their own licences and are downloaded separately.

| Component | Terms |
|---|---|
| [EyeClean](https://github.com/wessberg/eyeclean/blob/a3efafd6d2794cace9647251791f411bbeaf626f/LICENSE) | MIT code and r57 weights. Its Blender training tools have different terms; this pipeline doesn't use them. Detector weights have their own terms. |
| [CorridorKey 2.0](https://github.com/nikopueringer/CorridorKey2.0/blob/485f5e9ab759ead6138cc990776184b59d6ea971/LICENSE) | Corridor Key Licence v2.0, based on CC BY-NC-SA with additional terms. It permits commercial output and restricts redistribution, resale, paid inference services and integration. Organisations above its revenue threshold need plugin seats. Read the full licence. |
| [CorridorKey foundations](https://github.com/nikopueringer/CorridorKey2.0/blob/485f5e9ab759ead6138cc990776184b59d6ea971/THIRD_PARTY_NOTICES.md) | C-RADIO: NVIDIA Open Model Licence. MoGe: MIT with DINOv2 notices. RVM: Apache-2.0 code and CC-BY-4.0 weights. |
| [Sapiens2 normals](https://huggingface.co/facebook/sapiens2-normal-1b) | Sapiens2 model licence. |
| [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) | Check the chosen checkpoint's terms. This pipeline uses Small. |
| [Blender](https://www.blender.org/about/license/) | Installed separately and called as a process. |
| [FFmpeg](https://ffmpeg.org/legal.html) | Terms depend on the build. This project doesn't distribute its binary. |

Older experiments use NVIDIA DiffusionRenderer and other non-commercial models. The current
keying, EyeClean, normals and skin-shading path doesn't require them.

The shot metadata identifies the reporter footage. The package and source export exclude it.
Check the original provider's licence before publishing stills or clips.
