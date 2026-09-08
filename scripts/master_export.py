"""Editorial masters for DaVinci Resolve: ProRes 4444 or DNxHR 444, 4:4:4 10-bit PQ.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/master_export.py --shot <dir>
    ... --codec prores4444 | prores4444xq | dnxhr444 | all

The HEVC path in ``hdr_mux.py`` produces a *delivery* file: 4:2:0, 10-bit, heavily compressed,
correct for playback and wrong for grading. This produces the file that goes into Resolve --
all-intra, 4:4:4, no chroma subsampling, no inter-frame prediction to smear a grade across cuts.

Both encoders read the same scene-linear EXR sequence and the same transform chain as the HEVC
master (``color_pipeline``: Rec.709 linear -> Rec.2020 -> 1000-nit PQ), so the three outputs are
the same picture at different quantisations rather than three separate grades.

**10-bit is the ceiling here, and it is ffmpeg's, not the format's.** ProRes 4444 and DNxHR 444
both define 12-bit, but ffmpeg's encoders advertise only ``yuv444p10le`` / ``gbrp10le`` (verified
against the bundled build, not assumed). If 12 bits genuinely matter, do not use a video codec at
all: hand Resolve ``intermediate/linear_4k/*.exr`` directly. Those are 16-bit half float,
scene-linear and unquantised -- strictly better than any ProRes, and Resolve imports EXR
sequences natively. ``--report`` prints that path.

The mov timescale trap from the HEVC path applies here too and is handled the same way: the
timescale is the frame rate's numerator, so 60000/1001 is representable exactly and no edit list
appears to hide frames.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse, subprocess, shutil
from fractions import Fraction
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args
from color_pipeline import REC709_TO_2020, grade_nits, pq_encode

CODECS = {
    'prores4444':   {'args': ['-c:v', 'prores_ks', '-profile:v', '4', '-vendor', 'apl0',
                              '-bits_per_mb', '8000', '-pix_fmt', 'yuv444p10le'],
                     'ext': 'mov', 'label': 'Apple ProRes 4444, 4:4:4 10-bit'},
    'prores4444xq': {'args': ['-c:v', 'prores_ks', '-profile:v', '5', '-vendor', 'apl0',
                              '-bits_per_mb', '16000', '-pix_fmt', 'yuv444p10le'],
                     'ext': 'mov', 'label': 'Apple ProRes 4444 XQ, 4:4:4 10-bit'},
    'dnxhr444':     {'args': ['-c:v', 'dnxhd', '-profile:v', 'dnxhr_444',
                              '-pix_fmt', 'yuv444p10le'],
                     'ext': 'mov', 'label': 'Avid DNxHR 444, 4:4:4 10-bit'},
}


def ffmpeg_path():
    local = sorted(ROOT.glob('dependencies/ffmpeg/*/bin/ffmpeg.exe'))
    if local:
        return str(local[0])
    found = shutil.which('ffmpeg')
    if not found:
        raise SystemExit('No ffmpeg in dependencies/ffmpeg/*/bin or on PATH')
    return found


def encode(shot, codec, frames, exposure, out_dir, dry_run=False):
    spec = CODECS[codec]
    rate = shot.delivery_frame_rate
    w, h = shot.resolution
    out = out_dir / ('%s_%s.%s' % (shot.name, codec, spec['ext']))
    cmd = [ffmpeg_path(), '-y', '-hide_banner', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-s', '%dx%d' % (w, h),
           '-r', '%d/%d' % (rate.numerator, rate.denominator), '-i', 'pipe:0',
           *spec['args'],
           # Tagging only: the pixels are already Rec.2020 PQ, so nothing is converted here.
           '-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
           '-colorspace', 'bt2020nc', '-color_range', 'tv',
           # The round-1 lesson: a timescale that cannot represent the frame duration pads the
           # track and hides frames behind an edit list.
           '-video_track_timescale', str(rate.numerator),
           str(out)]
    if dry_run:
        return {'codec': codec, 'output': str(out), 'command': ' '.join(cmd), 'skipped': 'dry run'}

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    peak_nits, frame_avg = 0.0, []
    for i in range(frames):
        src = shot.master_exr(i)
        if not src.exists():
            proc.stdin.close(); proc.wait()
            raise SystemExit('Missing master frame %s. Run composite_shot.py --final first.' % src)
        lin = cv2.imread(str(src), -1)[..., ::-1].astype(np.float32)      # BGR -> RGB
        if (lin.shape[1], lin.shape[0]) != (w, h):
            lin = cv2.resize(lin, (w, h), interpolation=cv2.INTER_AREA)
        nits = grade_nits(lin @ REC709_TO_2020.T, exposure)
        peak_nits = max(peak_nits, float(nits.max()))
        frame_avg.append(float(np.mean(np.max(nits, axis=-1))))
        code = np.clip(pq_encode(nits), 0, 1)
        proc.stdin.write((np.rint(code * 65535).astype('<u2')).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit('%s encode failed' % codec)
    return {'codec': codec, 'label': spec['label'], 'output': str(out),
            'size_bytes': out.stat().st_size,
            'size_mb': round(out.stat().st_size / 1e6, 1),
            'max_cll_nits': round(peak_nits, 1),
            'max_fall_nits': round(float(np.mean(frame_avg)), 1),
            'command': ' '.join(cmd)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--codec', default='prores4444',
                    choices=list(CODECS) + ['all'])
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--exposure', type=float, default=1.0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--report', action='store_true',
                    help='describe the available handoffs and exit')
    args = ap.parse_args()
    shot = from_args(args)

    if args.report:
        exrs = sorted(shot.linear_4k.glob('*.exr'))
        print(json.dumps({
            'shot': shot.name,
            'best_fidelity_handoff': {
                'path': str(shot.linear_4k), 'frames': len(exrs),
                'format': 'OpenEXR half float, scene-linear Rec.709, unquantised',
                'why': 'strictly better than any 10-bit video codec; Resolve imports EXR '
                       'sequences natively. Set the clip to Linear / Rec.709 on import.'},
            'video_masters': {k: v['label'] for k, v in CODECS.items()},
            'ffmpeg_bit_depth_ceiling': '10-bit; ffmpeg advertises no 12-bit ProRes or DNxHR '
                                        'encoder, verified against the bundled build',
            'delivery_file': 'hdr_mux.py writes the HEVC 4:2:0 playback master separately'},
            indent=2))
        return

    frames = args.frames or shot.frame_count
    out_dir = shot.dir / 'master'
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = list(CODECS) if args.codec == 'all' else [args.codec]

    results = []
    for c in todo:
        print('encoding %s ...' % c, flush=True)
        results.append(encode(shot, c, frames, args.exposure, out_dir, args.dry_run))
        print(json.dumps(results[-1], indent=2), flush=True)

    summary = {'shot': shot.name, 'frames': frames,
               'resolution': list(shot.resolution),
               'frame_rate': str(shot.delivery_frame_rate),
               'transfer': 'Rec.2020 primaries, ST2084 PQ, 1000-nit mastering peak',
               'chroma': '4:4:4', 'bit_depth': 10,
               'bit_depth_note': 'ffmpeg has no 12-bit ProRes or DNxHR encoder. For more than '
                                 '10 bits, take intermediate/linear_4k/*.exr into Resolve.',
               'exports': results}
    shot.log('master_export.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
