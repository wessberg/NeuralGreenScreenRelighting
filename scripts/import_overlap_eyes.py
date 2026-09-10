"""Carry an accepted eye run and its tracks from one shot to another shot of the SAME plate.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/import_overlap_eyes.py \
        --shot music_videos/reporter_test/shots/reporter_a \
        --from music_videos/reporter_test/shots/reporter_full --run r40_floorB \
        --tracks limbus_track_tv.json,eyelid_track_wtv40.json

`reporter_a` (150 frames, where the proxy, the bulb tour and the ratio composite live) is
source frames 700-849 of `reporter_full` (1200 frames, where the accepted eye removal
`r40_floorB` lives). Round 41 needs the ACCEPTED removal pixels under the tour, not a fresh
150-frame run of the same pass: a re-run pools its canonical iris and its lid curvature over
a different clip length and would not be the frames the artist judged.

So this copies, never recomputes. The offset is read from both shots' `shot.json`
(`selection.start_frame`), and every copied frame is gated on the keyed foreground being
the SAME FILE: the run's `removal.json` records the sha256 of the fg EXR each frame was
built from, and the destination shot's fg for the rebased index must hash identically, or
the frame is refused. On these two shots all 150 hashes match (the key was run twice with
the same thresholds and produced byte-identical EXRs).

Writes, in the destination shot:

* `composite/<run>_import/{source,removal,weight}/<local>.png` -- the run's frames, renumbered;
* `composite/<run>_import/removal.json` -- the source run's rows for the overlap, renumbered,
  with the provenance block;
* `relight/<track>` for each track named, renumbered (`frame` rebased, `source_frame` kept),
  with the source path and its sha256 recorded in the metadata.

Pure copying and renumbering; no pixel is changed.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shot_paths import Shot, add_shot_argument, from_args


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--from', dest='src', required=True, help='shot dir holding the run')
    ap.add_argument('--run', required=True, help='composite/<run> in the source shot')
    ap.add_argument('--tracks', default='limbus_track_tv.json,eyelid_track_wtv40.json',
                    help='comma-separated relight/*.json tracks to rebase')
    ap.add_argument('--suffix', default='_import')
    args = ap.parse_args()
    dst = from_args(args)
    src = Shot(args.src)
    off_src = int((src.config.get('selection') or {}).get('start_frame', 0))
    off_dst = int((dst.config.get('selection') or {}).get('start_frame', 0))
    offset = off_dst - off_src
    n_dst = dst.frame_count
    print('offset: %s local i == %s local i + %d' % (dst.name, src.name, offset), flush=True)

    run_dir = src.composite / args.run
    rec = json.loads((run_dir / 'removal.json').read_text(encoding='utf-8'))
    rows_by_frame = {int(r['frame']): r for r in rec['rows']}
    out_dir = dst.composite / (args.run + args.suffix)
    for sub in ('source', 'removal', 'weight'):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    copied, rows = [], []
    for i in range(n_dst):
        j = i + offset
        row = rows_by_frame.get(j)
        if row is None:
            raise SystemExit('%s has no row for frame %d (dest %d)' % (args.run, j, i))
        fg_sha = sha(dst.fg(i))
        if row.get('source_sha256') != fg_sha:
            raise SystemExit('frame %d: dest fg sha %s != run source sha %s -- not the same '
                             'photograph, refusing' % (i, fg_sha[:12],
                                                       str(row.get('source_sha256'))[:12]))
        for sub in ('source', 'removal', 'weight'):
            shutil.copyfile(run_dir / sub / ('%06d.png' % j), out_dir / sub / ('%06d.png' % i))
        r2 = dict(row)
        r2['frame'] = i
        r2['imported_from_frame'] = j
        rows.append(r2)
        copied.append(i)
    meta = {k: v for k, v in rec.items() if k != 'rows'}
    meta['rows'] = rows
    meta['imported'] = {'from_shot': src.name, 'run': args.run, 'frame_offset': offset,
                        'frames': len(copied), 'gate': 'fg EXR sha256 per frame, all equal',
                        'source_removal_json_sha256': sha(run_dir / 'removal.json')}
    (out_dir / 'removal.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print('frames copied:', len(copied), '->', out_dir, flush=True)

    for name in [t.strip() for t in args.tracks.split(',') if t.strip()]:
        p = src.relight / name
        data = json.loads(p.read_text(encoding='utf-8'))
        out_rows = []
        for row in data['rows']:
            f = int(row['frame']) - offset
            if 0 <= f < n_dst:
                r2 = dict(row)
                r2['frame'] = f
                r2['imported_from_frame'] = int(row['frame'])
                out_rows.append(r2)
        if len(out_rows) != n_dst:
            raise SystemExit('%s covers %d of %d dest frames' % (name, len(out_rows), n_dst))
        out = {k: v for k, v in data.items() if k != 'rows'}
        out['frames'] = len(out_rows)
        out['imported'] = {'from_shot': src.name, 'path': str(p.relative_to(src.dir)),
                           'sha256': sha(p), 'frame_offset': offset}
        out['rows'] = out_rows
        stem = Path(name).stem + args.suffix + '.json'
        (dst.relight / stem).write_text(json.dumps(out), encoding='utf-8')
        print('track', name, '->', dst.relight / stem, flush=True)


if __name__ == '__main__':
    main()
