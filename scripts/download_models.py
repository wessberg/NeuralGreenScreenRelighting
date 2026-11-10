"""Download and verify the pinned CorridorKey 2.0 models; no model redistribution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ngsr.engines import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python with huggingface-hub installed")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))
    revision = manifest["models"]["corridorkey"]["revision"]
    subprocess.run([str(Path(args.python).resolve()), "-B",
        str(ROOT / "dependencies/CorridorKey2/scripts/download_weights.py"),
        "--corridorkey_revision", revision], check=True, cwd=ROOT)
    for name, model in manifest["models"].items():
        if "file" not in model:
            continue
        path = ROOT / model["file"]
        if file_sha256(path) != model["sha256"]:
            raise RuntimeError(f"Model SHA-256 mismatch: {name} ({path})")
        print(f"Verified {name}: {model['sha256']}")


if __name__ == "__main__":
    main()
