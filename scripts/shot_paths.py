"""Shot-agnostic path and configuration resolution.

Production scripts previously hardcoded project-root-relative paths (source/frames, key/,
relight/, intermediate/, composite/, final/, logs/) plus a literal 30 fps and 300 frames.
That tied the pipeline to the single proof-of-concept example. Scripts now accept
``--shot <dir>`` and read and write inside that shot directory. There is no default: the
proof of concept used to sit at the project root and now lives under music_videos/ like every
other shot, so omitting --shot lists the available shots instead of guessing.

Frame rate, frame count, keying thresholds and the delivery specification are read from the
shot's ``shot.json`` and its video's ``project.json``. Nothing here assumes a frame rate:
rates are exact ``Fraction`` values so fractional sources (60000/1001) stay exact.
"""
import json, argparse
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The proof-of-concept values, kept only as documented fallbacks for the legacy root layout.
# They are SHOT-SPECIFIC choices, never defaults for a new clip.
LEGACY_KEY_HINT = {"green_min": 0.68, "green_dominance_min": 0.48}


class Shot:
    """Resolves every input and output location for one shot."""

    @staticmethod
    def available():
        """Every shot directory in the workspace, as repo-relative paths."""
        return sorted(str(p.parent.relative_to(ROOT)).replace("\\", "/")
                      for p in (ROOT / "music_videos").glob("*/shots/*/shot.json"))

    def __init__(self, shot=None):
        if shot in (None, "", "."):
            # The proof of concept used to live at the project root. It now sits under
            # music_videos/ like every other shot, so there is no default any more.
            raise SystemExit("--shot is required. Available shots:\n  "
                             + "\n  ".join(self.available()))
        else:
            p = Path(shot)
            self.dir = p if p.is_absolute() else ROOT / p
        self.dir = self.dir.resolve()
        if not self.dir.is_dir():
            raise SystemExit("No such shot directory: %s" % self.dir)
        if ROOT not in self.dir.parents and self.dir != ROOT:
            raise SystemExit("Shot directory must live inside the project: %s" % self.dir)
        self.is_legacy_root = self.dir == ROOT
        self.name = self.dir.name
        self.config = self._load(self.dir / "shot.json")
        self.video_dir = self.dir.parents[1]
        self.video_config = self._load(self.video_dir / "project.json")

    @staticmethod
    def _load(path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    # ---- directories -------------------------------------------------------
    @property
    def source(self):        return self.dir / "source"
    @property
    def source_frames(self): return self.dir / "source/frames"
    @property
    def plate(self):         return self.dir / "source/plate.mp4"
    @property
    def key(self):           return self.dir / "key"
    @property
    def key_alpha(self):     return self.dir / "key/alpha"
    @property
    def key_fg(self):        return self.dir / "key/fg"
    @property
    def key_qc(self):        return self.dir / "key/qc"
    @property
    def relight(self):       return self.dir / "relight"
    @property
    def depth(self):         return self.dir / "relight/depth"
    @property
    def controls(self):      return self.dir / "relight/controls"
    @property
    def composite(self):     return self.dir / "composite"
    @property
    def intermediate(self):  return self.dir / "intermediate"
    @property
    def linear_4k(self):     return self.dir / "intermediate/linear_4k"
    @property
    def final(self):         return self.dir / "final"
    @property
    def logs(self):          return self.dir / "logs"

    def log(self, name):     return self.logs / name

    # Environments live in scene_library/ and are shared between shots. Each shot names its
    # own via shot.json {"environment": {"dir": "scene_library/<name>"}}.
    @property
    def environment(self):
        named = (self.config.get("environment") or {}).get("dir")
        if not named:
            raise SystemExit(
                "%s does not name an environment; set environment.dir, e.g. "
                "scene_library/night_restaurant" % (self.dir / "shot.json"))
        return ROOT / named

    @property
    def lighting_json(self):
        return self.environment / "lighting.json"

    def background(self, test=False):
        return self.environment / ("test_background_shadow.exr" if test else "background_shadow.exr")

    def ensure(self):
        for d in (self.source_frames, self.key_alpha, self.key_fg, self.key_qc,
                  self.depth, self.controls, self.composite, self.linear_4k,
                  self.final, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # ---- timing ------------------------------------------------------------
    @property
    def frame_rate(self):
        """Exact source frame rate. Never guessed: absent means the shot is not described."""
        r = self.config.get("source_frame_rate")
        if r is None:
            raise SystemExit(
                "%s has no source_frame_rate; a clip's rate must be recorded, not assumed"
                % (self.dir / "shot.json"))
        return Fraction(r)

    @property
    def frame_count(self):
        """Frames in the selected range: the recorded selection, else what was extracted."""
        sel = self.config.get("selection") or {}
        if isinstance(sel.get("frame_count"), int):
            return sel["frame_count"]
        n = len(list(self.source_frames.glob("*.png")))
        if not n:
            raise SystemExit(
                "No selection.frame_count in shot.json and no extracted frames in %s"
                % self.source_frames)
        return n

    @property
    def source_start_frame(self):
        return (self.config.get("selection") or {}).get("start_frame", 0)

    @property
    def duration_seconds(self):
        return self.frame_count / float(self.frame_rate)

    # ---- working resolution -----------------------------------------------
    # The estimator stages (normals, depth-derived geometry, albedo, SH relight) run at a
    # reduced resolution for speed. That size was hardcoded 768x432 in four scripts, which is
    # 16:9 -- correct only by accident, because the one shot in the workspace was 16:9. On a
    # 2.40:1 anamorphic plate it would have squashed every normal and every proxy vertex, and
    # the result would have looked plausible and been wrong.
    #
    # Derived from the PLATE's own aspect at a constant pixel budget, so it follows the source
    # from 1080p 16:9 to 5.8K 17:9 to a desqueezed 2.40:1 without anyone editing a constant.
    WORKING_PIXELS = 768 * 432

    @property
    def working_resolution(self):
        """(width, height) for the estimator stages, matching the plate's aspect ratio."""
        sw, sh = self.config.get("source_resolution", [1920, 1080])
        aspect = float(sw) / float(sh)
        w = int(round((self.WORKING_PIXELS * aspect) ** 0.5 / 4.0)) * 4
        h = int(round(w / aspect / 4.0)) * 4
        return max(w, 64), max(h, 64)

    @property
    def delivery(self):
        """Shot delivery overrides video delivery overrides source rate."""
        d = dict(self.video_config.get("delivery") or {})
        d.update(self.config.get("delivery") or {})
        return d

    @property
    def delivery_frame_rate(self):
        """Configurable output rate. Falls back to the source rate, meaning no conform."""
        r = self.delivery.get("frame_rate")
        return Fraction(r) if r else self.frame_rate

    @property
    def conform(self):
        """How the source rate becomes the delivery rate. Must be explicit when they differ."""
        policy = self.delivery.get("conform") or self.delivery.get("conform_decision")
        if self.delivery_frame_rate == self.frame_rate:
            return "preserve"
        if not policy or str(policy).upper().startswith("PENDING"):
            raise SystemExit(
                "%s: delivery rate %s differs from source %s but no conform policy is recorded. "
                "Retiming is a creative decision; set delivery.conform in shot.json "
                "(preserve | resample_nearest | resample_blend | retime_slowmo)."
                % (self.name, self.delivery_frame_rate, self.frame_rate))
        return policy

    @property
    def resolution(self):
        r = self.delivery.get("resolution")
        return tuple(r) if r else (3840, 2160)

    # ---- keying ------------------------------------------------------------
    @property
    def key_hint(self):
        """Screen-detection thresholds. Per-shot: backing brightness, saturation and whether
        the subject wears green all differ between clips."""
        h = dict(LEGACY_KEY_HINT)
        h.update((self.config.get("keying") or {}).get("hint") or {})
        return h

    # ---- per-frame files ---------------------------------------------------
    def frame(self, i):      return self.source_frames / ("%06d.png" % i)
    def alpha(self, i):      return self.key_alpha / ("%06d.exr" % i)
    def fg(self, i):         return self.key_fg / ("%06d.exr" % i)
    def depth_npy(self, i):  return self.depth / ("%06d.npy" % i)
    def control(self, i):    return self.controls / ("%06d.npz" % i)
    def master_exr(self, i): return self.linear_4k / ("%06d.exr" % i)

    def describe(self):
        return {"shot": self.name, "dir": str(self.dir),
                "source_frame_rate": str(self.frame_rate),
                "frame_count": self.frame_count,
                "source_start_frame": self.source_start_frame,
                "duration_seconds": round(self.duration_seconds, 6),
                "delivery_frame_rate": str(self.delivery_frame_rate),
                "conform": self.conform,
                "resolution": list(self.resolution),
                "environment": str(self.environment),
                "working_resolution": list(self.working_resolution),
                "key_hint": self.key_hint}


def add_shot_argument(parser):
    parser.add_argument("--shot", default=None, required=False,
                        help="Shot directory, e.g. music_videos/reporter_test/shots/reporter_a")
    return parser


def from_args(args):
    return Shot(getattr(args, "shot", None))


if __name__ == "__main__":
    parser = add_shot_argument(argparse.ArgumentParser(description="Show resolved shot configuration"))
    print(json.dumps(Shot(parser.parse_args().shot).describe(), indent=2))
