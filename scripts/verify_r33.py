"""Verify round 33: exact source outside the mask, plus the standing fallback checks.

Three things, none of which is an accuracy claim:

1. Every written frame is bit-exact source wherever the reported mask is zero, and every
   value is finite and non-negative.
2. The repaired region is genuinely darker than the source's highlight, per eye, so a
   "repair" that quietly did nothing cannot pass.
3. The standing 300 exact keyed-eye fallbacks and 304 protected hashes, reused from
   `verify_r15.fallback` unchanged.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/verify_r33.py \\
        --shot music_videos/reporter_test/shots/reporter_a
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shot_paths import add_shot_argument, from_args
from eye_reflection_repair import rgb8, sha, write_json
from verify_r15 import fallback


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--name', default='r33_repair')
    args = ap.parse_args()
    shot = from_args(args)
    root = shot.composite / args.name
    report = json.loads((root / 'repair.json').read_text())

    exact, checked, darker, not_darker = 0, 0, 0, []
    for row in report['rows']:
        i = row['frame']
        src = cv2.imread(str(root / 'source' / ('%06d.png' % i)))
        rep = cv2.imread(str(root / 'repair' / ('%06d.png' % i)))
        mask = cv2.imread(str(root / 'mask' / ('%06d.png' % i)), cv2.IMREAD_GRAYSCALE)
        assert src is not None and rep is not None and mask is not None, i
        raw = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        assert raw is not None and np.isfinite(raw).all() and (raw >= 0).all(), i
        # The written source frame must be the keyed plate itself, not a re-render.
        assert np.array_equal(src, rgb8(raw[..., ::-1].astype(np.float32))[..., ::-1]), i
        untouched = mask == 0
        assert np.array_equal(rep[untouched], src[untouched]), 'source changed outside mask %d' % i
        assert row['changed_pixels'] == int((mask > 0).sum()), i
        exact += int(untouched.sum())
        checked += 1
        for entry in row['eyes']:
            if not entry.get('repaired'):
                continue
            if entry['peak_repaired_display'] < entry['peak_source_display']:
                darker += 1
            else:
                not_darker.append((i, entry['side']))

    record = dict(
        frames_checked=checked,
        exact_source_pixels_outside_mask=exact,
        repaired_eyes_darker_than_source=darker,
        repaired_eyes_not_darker=not_darker,
        source_outside_mask_exact=True,
        **fallback(shot),
        repair_code_sha256=report['code_sha256'],
        atlas_code_sha256=report['atlas_code_sha256'],
        code_sha256=sha(__file__),
        production_enabled=False, artist_accepted=False)
    assert not not_darker, not_darker
    write_json(shot.log('r33_verification.json'), record)
    print(json.dumps({k: v for k, v in record.items() if k != 'repaired_eyes_not_darker'},
                     indent=1))


if __name__ == '__main__':
    main()
