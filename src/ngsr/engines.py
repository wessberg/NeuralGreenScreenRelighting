"""Lossless file contracts across incompatible model environments."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_eyeclean(foreground, output, *, fps, engine=None, model="eyeclean-r57",
                 max_frames=None, on_event=None, cancel=None):
    """Run EyeClean on straight linear Rec.709 EXRs, binding patches to source hashes."""
    from eyeclean_client import run
    foreground, output = Path(foreground).resolve(), Path(output).resolve()
    rate = Fraction(fps)
    if rate <= 0 or (max_frames is not None and max_frames <= 0):
        raise ValueError("fps and max_frames must be positive")
    paths = sorted(foreground.glob("*.exr"))
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise ValueError(f"No EXR foreground frames in {foreground}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh EyeClean output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if any(p.name != f"{int(p.stem):06d}.exr" for p in paths):
        raise ValueError("Foreground filenames must use six-digit frame indices")
    ids = [int(p.stem) for p in paths]
    if ids != list(range(ids[0], ids[0] + len(ids))):
        raise ValueError("EyeClean needs a contiguous sequence for temporal tracking")
    rows = [{"index": i, "frame": int(p.stem), "file": p.name,
             "sha256": file_sha256(p)} for i, p in enumerate(paths)]
    if len({r["frame"] for r in rows}) != len(rows):
        raise ValueError("Foreground filenames must have unique numeric stems")
    job = {"schema": "eyeclean.job/1", "input": str(foreground / "%06d.exr"),
           "output": str(output), "input_color": "linear-rec709",
           "working_color": "linear-rec709", "fps": str(rate), "model": model,
           "export": ["patches", "report"], "exr_compression": "zip",
           "client": {"product": "neural-green-screen-relighting", "contract": "ngsr.eyes/1"}}
    if max_frames is not None:
        job["max_frames"] = max_frames
    result = run(job, engine=engine, on_event=on_event, cancel=cancel,
                 stderr_path=str(output.parent / (output.name + ".stderr.log")))
    report = result.report
    preservation = report.get("preservation", {})
    if preservation.get("source_files_modified", 0) or preservation.get("outside_support_changed_bits", 0):
        raise ValueError("EyeClean preservation verification failed")
    if report["frames"] != len(rows):
        raise ValueError("EyeClean frame count differs from the submitted sequence")
    for row in rows:
        if file_sha256(foreground / row["file"]) != row["sha256"]:
            raise ValueError("Source changed during EyeClean processing")
    manifest = {"schema": "ngsr.eyes/1", "fps": str(rate), "frames": rows,
                "model": report["model"], "engine_version": report["eyeclean_version"]}
    (output / "ngsr.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def run_keyer(input_frames, output, *, repository, python, fps, ffmpeg="ffmpeg",
              max_frames=None, hint=None, low_vram=False):
    """Run official CorridorKey 2.0; return straight linear alpha/FG FLOAT EXRs.

    Inputs must be desqueezed, sRGB-transfer / Rec.709-primary decoded frames.
    """
    root, output = Path(repository).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh keyer output directory: {output}")
    if Fraction(fps) <= 0 or (max_frames is not None and max_frames <= 0):
        raise ValueError("fps and max_frames must be positive")
    command = [str(Path(python).resolve()), "-B", "-u", str(root / "infer.py"),
               "--input", str(Path(input_frames).resolve()), "--output_dir", str(output),
               "--frame_dir_fps", str(float(Fraction(fps))), "--outputs", "alpha", "fg",
               "--output_format", "exr", "--exr_dtype", "float", "--despill_strength", "0",
               "--ffmpeg_bin", str(ffmpeg), "--temp_dir", str(output / "scratch")]
    if max_frames is not None:
        command += ["--num_frames", str(max_frames)]
    if hint is not None:
        command += ["--hint_video", str(Path(hint).resolve())]
    if low_vram:
        command += ["--low_vram", "--hann_chunk", "8", "--hann_stride", "4"]
    subprocess.run(command, cwd=root, check=True)
    alpha, fg = sorted(output.glob("*_alpha_exr")), sorted(output.glob("*_fg_exr"))
    if len(alpha) != 1 or len(fg) != 1:
        raise RuntimeError(f"Unexpected CorridorKey output layout: {output}")
    (output / "ngsr.json").write_text(json.dumps({"schema": "ngsr.key/1",
        "fps": str(Fraction(fps)), "command": command, "alpha": alpha[0].name,
        "foreground": fg[0].name, "color": "linear-rec709", "alpha_mode": "straight"},
        indent=2), encoding="utf-8")
    return alpha[0], fg[0]
