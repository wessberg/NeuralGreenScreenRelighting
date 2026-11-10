"""Fetch the pinned upstream source interfaces; no environments or weights are installed."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))
    for name, source in manifest["sources"].items():
        directory = ROOT / source["directory"]
        if directory.exists():
            dirty = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"], text=True)
            if dirty.strip():
                raise SystemExit(f"Refusing to update modified dependency: {directory}")
            head = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
            if head == source["revision"]:
                print(f"{name}: already pinned")
                continue
        else:
            directory.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", source["url"], str(directory)], check=True)
        subprocess.run(["git", "-C", str(directory), "fetch", "origin", source["revision"]], check=True)
        subprocess.run(["git", "-C", str(directory), "checkout", "--detach", source["revision"]], check=True)
        print(f"{name}: {source['revision']}")


if __name__ == "__main__":
    main()
