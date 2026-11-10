"""Run EyeClean out of process on a shot's straight linear Rec.709 foreground."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ngsr.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "eyes")
    main()
