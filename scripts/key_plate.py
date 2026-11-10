"""CorridorKey 2.0 keying with original-core preservation and per-shot despill."""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ngsr import run_keyer
from ngsr.keying import hint_for, preserve_core
from shot_paths import add_shot_argument, from_args


def read_source(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB source frame: {path}")
    scale = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1
    return np.ascontiguousarray(image[..., ::-1].astype(np.float32) / scale)


def main():
    parser = add_shot_argument(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--repository", type=Path, default=ROOT / "dependencies/CorridorKey2")
    parser.add_argument("--python", type=Path, default=None, help="Isolated CorridorKey 2.0 Python")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--test", action="store_true", help="First 8 contiguous frames (temporal model)")
    parser.add_argument("--low-vram", action="store_true")
    args = parser.parse_args()
    shot = from_args(args)
    count = min(8, shot.frame_count) if args.test else (args.frames or shot.frame_count)
    if not 0 < count <= shot.frame_count:
        parser.error("--frames must be within the shot")
    if any(shot.key_fg.glob("*.exr")) or any(shot.key_alpha.glob("*.exr")):
        raise SystemExit("Key already exists. Use a fresh shot/output directory to compare versions.")
    python = args.python or args.repository / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    raw = shot.key / "corridorkey2"
    hint_dir = shot.key / "hints"
    hint_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        hint = hint_for(read_source(shot.frame(i)), shot.key_hint)
        if not cv2.imwrite(str(hint_dir / f"{i:06d}.png"), np.rint(hint*65535).astype(np.uint16)):
            raise RuntimeError("Cannot write key hint")
    alpha_dir, fg_dir = run_keyer(shot.source_frames, raw, repository=args.repository,
        python=python, fps=shot.frame_rate, ffmpeg=args.ffmpeg, max_frames=count,
        hint=hint_dir, low_vram=args.low_vram)
    for directory in (shot.key_fg, shot.key_alpha, shot.key_qc, shot.logs):
        directory.mkdir(parents=True, exist_ok=True)
    flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT,
             cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP]
    rows = []
    for i in range(count):
        a = cv2.imread(str(alpha_dir / f"alpha_{i:06d}.exr"), -1)
        fg = cv2.imread(str(fg_dir / f"fg_{i:06d}.exr"), -1)
        if a is None or fg is None:
            raise RuntimeError(f"Missing CorridorKey result {i}")
        if a.ndim == 3:
            a = a[..., 0]
        source = read_source(shot.frame(i))
        hint = cv2.imread(str(hint_dir / f"{i:06d}.png"), -1).astype(np.float32)/65535
        fg, a = preserve_core(source, fg[..., ::-1], a, shot.config.get("keying", {}), hint)
        for path, value in ((shot.fg(i), fg[..., ::-1]), (shot.alpha(i), a)):
            if not cv2.imwrite(str(path), value, flags):
                raise RuntimeError(f"Cannot write {path}")
        rows.append({"frame": i, "opaque_pixels": int((a >= .98).sum())})
    report = {"schema": "ngsr.key/1", "backend": "CorridorKey 2.0",
              "frames": count, "fps": str(shot.frame_rate), "color": "linear-rec709",
              "alpha_mode": "straight", "keying": shot.config.get("keying", {}), "per_frame": rows}
    shot.log("key_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("backend", "frames", "fps")}), flush=True)


if __name__ == "__main__":
    main()
