"""Verify round 34: exact source outside the weight, and the reflection only in the eye.

Five things, none of which is an accuracy claim:

1. Every written removal frame is bit-exact source wherever the reported weight is zero,
   and every value is finite and non-negative.
2. The written source frame is the keyed plate itself, not a re-render of it.
3. The removal DARKENS: the peak inside the confident region falls, per eye, so a
   "removal" that quietly did nothing cannot pass.
4. **The added reflection reaches nothing outside the eyes.** This is the check stage 11d
   did not have: its catchlight cap was a flat disc with no eyelid intersection and it
   painted silver arcs across the upper lids. Every pixel where the replaced frame
   differs from the removal frame must lie inside a tracked eyelid polygon, and the count
   outside is reported rather than asserted away.
5. The standing 300 exact keyed-eye fallbacks and 304 protected hashes, reused from
   `verify_r15.fallback` unchanged.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/verify_r34.py \\
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
import eye_reflection_repair as r32
from eye_reflection_repair import rgb8, sha, write_json
from verify_r15 import fallback


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--name', default='r34_removal')
    args = ap.parse_args()
    shot = from_args(args)
    root = shot.composite / args.name
    report = json.loads((root / 'removal.json').read_text())
    has_replace = (root / 'replace').is_dir()

    topology = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        points, found = z['points'], z['found']

    exact, checked, darker, not_darker = 0, 0, 0, []
    replaced_frames, add_inside, add_outside = 0, 0, 0
    for row in report['rows']:
        i = row['frame']
        src = cv2.imread(str(root / 'source' / ('%06d.png' % i)))
        rem = cv2.imread(str(root / 'removal' / ('%06d.png' % i)))
        weight = cv2.imread(str(root / 'weight' / ('%06d.png' % i)), cv2.IMREAD_GRAYSCALE)
        assert src is not None and rem is not None and weight is not None, i
        raw = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        assert raw is not None and np.isfinite(raw).all() and (raw >= 0).all(), i
        assert np.array_equal(src, rgb8(raw[..., ::-1].astype(np.float32))[..., ::-1]), i
        untouched = weight == 0
        assert np.array_equal(rem[untouched], src[untouched]), \
            'source changed outside the weight, frame %d' % i
        assert row['changed_pixels'] == int((weight > 0).sum()), i
        exact += int(untouched.sum())
        checked += 1
        for entry in row['eyes']:
            if not entry.get('repaired'):
                continue
            if entry['peak_removed_display'] < entry['peak_source_display']:
                darker += 1
            else:
                not_darker.append((i, entry['side']))
        if has_replace:
            rep = cv2.imread(str(root / 'replace' / ('%06d.png' % i)))
            if rep is None:
                continue
            replaced_frames += 1
            diff = np.any(rep != rem, axis=-1)
            if not diff.any():
                continue
            # The union of both eyelid polygons, rasterized where the eyes actually are.
            allowed = np.zeros(diff.shape, bool)
            if found[i] and np.isfinite(points[i]).all():
                for side in r32.SIDES:
                    geom, _ = r32.eye_frame(raw[..., ::-1].astype(np.float32), points[i],
                                            topology, side)
                    if geom is None:
                        continue
                    win = geom['window']
                    lid = r32.lid_distance(points[i], topology['eyelid_' + side], win)
                    allowed[win] |= lid > 0
            add_inside += int((diff & allowed).sum())
            add_outside += int((diff & ~allowed).sum())
        print('R34_VERIFY frame', i, flush=True)

    # The 300 fallbacks and 304 protected hashes are reporter_a's own fixtures (rounds 10
    # and 13). A shot without them -- reporter_full, the whole clip -- says so in the record
    # rather than skipping it silently or failing checks 1-4 that do apply to it.
    fixtures = shot.composite / 'r10_iris/correspondence_detail.json'
    standing = (fallback(shot) if fixtures.exists() else
                dict(exact_keyed_eye_fallbacks=None, protected_hashes_unchanged=None,
                     fallback_fixtures='absent on this shot (%s); not checked' % fixtures))
    record = dict(
        frames_checked=checked,
        exact_source_pixels_outside_weight=exact,
        removed_eyes_darker_than_source=darker,
        removed_eyes_not_darker=not_darker,
        source_outside_weight_exact=True,
        replace_frames_checked=replaced_frames,
        reflection_pixels_inside_eyelid=add_inside,
        reflection_pixels_outside_eyelid=add_outside,
        **standing,
        removal_code_sha256=report['code_sha256'],
        atlas_code_sha256=report['atlas_code_sha256'],
        code_sha256=sha(__file__),
        production_enabled=False, artist_accepted=False)
    assert not not_darker, not_darker
    assert add_outside == 0, 'reflection reached %d pixels outside the eyelids' % add_outside
    write_json(shot.log('r34_verification.json'), record)
    print(json.dumps({k: v for k, v in record.items() if k != 'removed_eyes_not_darker'},
                     indent=1))


if __name__ == '__main__':
    main()
