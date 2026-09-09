"""Prepare source-only native eye boards; never load predicted contours."""
import argparse
import json
import numpy as np
import cv2
from audit_r24_source import write, sha, native_hash, ROOT
from color_pipeline import linear_to_srgb
from shot_paths import add_shot_argument, from_args

FRAMES = (30, 60, 90, 120)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    args = ap.parse_args(); shot = from_args(args)
    root = shot.relight/'neural_iris_observations'
    manifest = json.loads((root/'manifest.json').read_text())
    out = shot.composite/'r25_source'; out.mkdir(parents=True, exist_ok=True)
    records = []
    for frame in FRAMES:
        panels = []
        for side in ('left', 'right'):
            path = root/f'{frame:06d}_{side}.npz'
            with np.load(path) as d:
                rgb, origin = d['linear_rgb'], d['origin_xy']
            raw = np.clip(linear_to_srgb(rgb)[..., ::-1]*255, 0, 255).astype('uint8')
            h, w = raw.shape[:2]
            board = np.zeros((h*8+50, w*8, 3), 'uint8')
            board[50:] = cv2.resize(raw, (w*8, h*8), interpolation=cv2.INTER_NEAREST)
            cv2.putText(board, f'{side} source {manifest["frames"][frame]["source_frame"]} native {w}x{h}',
                        (5,20), 0, .5, (255,255,255), 1)
            for x in range(0,w,5):
                cv2.putText(board, str(x), (x*8,44), 0,.3,(0,255,255),1)
            for y in range(0,h,5):
                cv2.putText(board, str(y), (0,50+y*8+7),0,.3,(0,255,255),1)
            file = out/f'source_{frame:06d}_{side}.png'
            assert cv2.imwrite(str(file), board)
            records.append(dict(frame=frame, side=side, source_frame=manifest['frames'][frame]['source_frame'],
                archive_sha256=sha(path), rgb_sha256=native_hash(rgb), origin_xy=origin.tolist(), shape=list(rgb.shape)))
    report = dict(records=records, protocol_sha256=sha(ROOT/'docs/r25-eye-protocol.md'),
                  neural_contours_loaded=False, annotations_complete=False)
    path = shot.logs/'r25_source_preflight.json'
    if path.exists(): assert json.loads(path.read_text()) == report
    else: write(path, report)
    print('R25_SOURCE_PREPARED')


if __name__ == '__main__': main()
