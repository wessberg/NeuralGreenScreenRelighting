"""Call established geometry and compositor stages from another Python."""
from pathlib import Path
import subprocess
import sys

STAGES = {
    "key": "key_plate.py",
    "normals": "neural_normals.py", "validate-normals": "validate_normals_convention.py",
    "depth": "video_depth.py", "track": "face_track.py",
    "source-light": "estimate_source_light.py", "albedo": "make_albedo.py",
    "proxy": "make_proxy_mesh.py", "eye-geometry": "eye_geometry.py",
    "limbus": "limbus_fit.py", "lids": "eyelid_fit.py",
    "render-tour": "stress_proxy.py", "render-eyes": "render_eye_catchlight.py",
    "sclera": "sclera_glint_removal.py",
    "composite": "composite_ratio.py", "review": "review_video.py", "master": "master_export.py",
}
BLENDER_STAGES = {"render-tour", "render-eyes"}


def run_stage(stage, shot, *, repository, python=sys.executable, blender=None, options=()):
    """Run a stage with an argv list, raising on failure. Requires a source checkout."""
    root = Path(repository).resolve()
    script = root / "scripts" / STAGES[stage]
    if not script.is_file():
        raise FileNotFoundError(f"Stage requires the source checkout: {script}")
    if stage in BLENDER_STAGES:
        if blender is None:
            raise ValueError(f"{stage} requires a Blender executable")
        command = [str(blender), "-b", "--factory-startup", "--python", str(script), "--"]
    else:
        command = [str(python), "-B", str(script)]
    command += ["--shot", str(Path(shot).resolve()), *map(str, options)]
    return subprocess.run(command, cwd=root, check=True)
