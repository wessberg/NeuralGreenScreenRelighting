import argparse
import json
import sys
from pathlib import Path
from . import run_eyeclean, run_stage
from .stages import STAGES


def main():
    parser = argparse.ArgumentParser(description="Neural green-screen actor relighting")
    commands = parser.add_subparsers(dest="command", required=True)
    eyes = commands.add_parser("eyes", help="Run EyeClean on keyed linear EXRs")
    eyes.add_argument("--shot", type=Path, required=True)
    eyes.add_argument("--output", type=Path, required=True)
    eyes.add_argument("--model", default="eyeclean-r57")
    eyes.add_argument("--max-frames", type=int)
    stage = commands.add_parser("stage", help="Run a geometry, render or composition stage")
    stage.add_argument("stage", choices=sorted(STAGES))
    stage.add_argument("--shot", type=Path, required=True)
    stage.add_argument("--repository", type=Path, default=Path.cwd())
    stage.add_argument("--python", default=sys.executable)
    stage.add_argument("--blender")
    args, extra = parser.parse_known_args()
    if args.command == "eyes":
        if extra:
            parser.error(f"Unknown arguments: {extra}")
        meta = json.loads((args.shot / "shot.json").read_text(encoding="utf-8"))
        run_eyeclean(args.shot / "key/fg", args.output, fps=meta["source_frame_rate"],
                     model=args.model, max_frames=args.max_frames,
                     on_event=lambda event: print(json.dumps(event), flush=True))
    else:
        run_stage(args.stage, args.shot, repository=args.repository, python=args.python,
                  blender=args.blender, options=extra[1:] if extra[:1] == ["--"] else extra)


if __name__ == "__main__":
    main()
