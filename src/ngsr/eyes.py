"""Apply float corrections to the exact keyed plate they were made from."""
import json
from pathlib import Path
from .engines import file_sha256


class EyeCleanPatches:
    def __init__(self, directory):
        from eyeclean_client import RunOutput
        self.path = Path(directory).resolve()
        self.run = RunOutput(directory)
        if self.run.index.get("schema") != "eyeclean.process/1":
            raise ValueError("Unsupported EyeClean index schema")
        if self.run.working_color != "linear-rec709":
            raise ValueError("EyeClean patches must be linear-rec709")
        binding = json.loads((self.path / "ngsr.json").read_text(encoding="utf-8"))
        if binding.get("schema") != "ngsr.eyes/1":
            raise ValueError("Unsupported NGSR eye binding")
        frames = {fr.frame: fr for fr in self.run.frames()}
        self.frames = {r["frame"]: (r, frames[r["index"]]) for r in binding["frames"]}

    def apply(self, foreground_rgb, frame, source_file):
        """Return float32 RGB, repaired subject-eye names, and changed pixel count.

        CUDA additions preserve all untouched bits, including signed zero. Stale
        sources, wrong dimensions and missing frames fail explicitly.
        """
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("EyeClean patch compositing requires CUDA")
        if frame not in self.frames:
            raise ValueError(f"EyeClean run has no source frame {frame}")
        binding, record = self.frames[frame]
        if file_sha256(source_file) != binding["sha256"]:
            raise ValueError(f"Stale EyeClean patches for frame {frame}; rerun after keying")
        width, height = self.run.size
        if foreground_rgb.shape != (height, width, 3) or foreground_rgb.dtype != np.float32:
            raise ValueError("Foreground must match the run's size and be float32 RGB")
        image = torch.as_tensor(foreground_rgb.copy(), device="cuda")
        changed = torch.zeros((height, width), dtype=torch.bool, device="cuda")
        sides = set()
        for eye in record.eyes:
            if eye.status != "repaired":
                continue
            if eye.eye not in ("image_right", "image_left"):
                raise ValueError(f"Unknown EyeClean eye convention: {eye.eye}")
            x0, y0, x1, y1 = eye.data_window
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError("EyeClean data window is outside the foreground")
            if not (self.path / eye.patch).resolve().is_relative_to(self.path):
                raise ValueError("EyeClean patch path escapes the run")
            channels = self.run.read(eye.patch).channels
            patch = np.stack([channels[c] for c in "RGB"], axis=-1).astype(np.float32)
            if patch.shape != (y1-y0, x1-x0, 3) or not np.isfinite(patch).all():
                raise ValueError("Invalid EyeClean correction")
            correction = torch.as_tensor(patch, device="cuda")
            support = torch.any(correction != 0, dim=-1)
            window = image[y0:y1, x0:x1]
            window[support] = window[support] + correction[support]
            changed[y0:y1, x0:x1] |= support
            sides.add("left" if eye.eye == "image_right" else "right")
        return image.cpu().numpy(), sides, int(changed.sum().item())
