"""Prepare frozen source-only boards for the round-27 diagnostic."""
import argparse
import json
import cv2
import numpy as np
from audit_r24_source import ROOT, sha, write, native_hash
from color_pipeline import linear_to_srgb
from shot_paths import add_shot_argument, from_args


def main():
    ap = argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    shot = from_args(ap.parse_args())
    out = shot.composite/'r27_photometry'; out.mkdir(parents=True, exist_ok=True)
    records = []
    for side in ('left', 'right'):
        path = shot.relight/f'neural_iris_observations/000115_{side}.npz'
        with np.load(path) as data:
            rgb, origin = data['linear_rgb'], data['origin_xy']
        raw = np.clip(linear_to_srgb(rgb)[..., ::-1]*255, 0, 255).astype('uint8')
        h, w = raw.shape[:2]; board = np.zeros((h*10+50, w*10, 3), 'uint8')
        board[50:] = cv2.resize(raw, (w*10, h*10), interpolation=cv2.INTER_NEAREST)
        for x in range(0, w, 5): cv2.putText(board, str(x), (x*10, 42), 0, .4, (0,255,255), 1)
        for y in range(0, h, 5): cv2.putText(board, str(y), (0, 50+y*10+8), 0, .4, (0,255,255), 1)
        cv2.putText(board, side+' source 815 native RGB', (5,20), 0,.6,(255,255,255),1)
        assert cv2.imwrite(str(out/f'source_115_{side}.png'), board)
        records.append(dict(side=side, archive_sha256=sha(path), rgb_sha256=native_hash(rgb),
                            origin_xy=origin.tolist(), shape=list(rgb.shape)))
    record = dict(records=records, protocol_sha256=sha(ROOT/'docs/r27-eye-protocol.md'))
    path = shot.logs/'r27_preflight.json'
    if path.exists(): assert json.loads(path.read_text()) == record
    else: write(path, record)
    print('R27_SOURCE_PREPARED')


if __name__ == '__main__': main()
