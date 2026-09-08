"""Editorial masters for DaVinci Resolve: ProRes 4444 or DNxHR 444, 4:4:4 10-bit PQ.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/master_export.py --shot <dir>
    ... --codec prores4444 | prores4444xq | dnxhr444 | all

The HEVC path in ``hdr_mux.py`` produces a *delivery* file: 4:2:0, 10-bit, heavily compressed,
correct for playback and wrong for grading. This produces the file that goes into Resolve --
all-intra, 4:4:4, no chroma subsampling, no inter-frame prediction to smear a grade across cuts.

Both encoders read the same scene-linear EXR sequence and the same transform chain as the HEVC
master (``color_pipeline``: Rec.709 linear -> Rec.2020 -> 1000-nit PQ), so the three outputs are
the same picture at different quantisations rather than three separate grades.

**The written file is a 12-bit ProRes 4444 carrying 10 bits of real data**, and that limit is
ffmpeg's rather than the format's: ProRes 4444 and DNxHR 444 both define 12-bit, ffprobe reports
the output as ``yuv444p12le``, but ffmpeg's *encoders* accept only ``yuv444p10le`` / ``gbrp10le``
(verified against the bundled build, not assumed), so the samples are quantised to 10 bits on the
way in. If 12 bits genuinely matter, do not use a video codec at all: hand Resolve
``intermediate/linear_4k/*.exr`` directly. Those are 16-bit half float, scene-linear and
unquantised -- strictly better than any ProRes, and Resolve imports EXR sequences natively.
``--report`` prints that path.

**Colour tagging needs more than the obvious flags, and the two codecs fail differently.**
Measured by probing the written file, which is the only way this class of bug surfaces:

===================================  ==========================================
 encoder settings                     resulting tags
===================================  ==========================================
 prores + bsf, no ``write_colr``      bt2020 / smpte2084 / bt2020nc  -- correct
 prores + bsf + ``write_colr``        unknown / unknown / bt2020nc
 prores, no bsf                       unknown / unknown / bt2020nc
 dnxhr, no ``write_colr``             unknown / unknown / **bt709**
 dnxhr + ``write_colr``               unknown / unknown / bt2020nc
===================================  ==========================================

So ProRes needs the ``prores_metadata`` bitstream filter and must NOT get ``+write_colr``, which
downgrades the colr atom to a legacy form that cannot express PQ. **DNxHR cannot be fully tagged
at all here** -- prefer ``prores4444``, or set the colour space by hand on import. Left untagged,
Resolve reads a PQ HDR master as SDR.

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

# The ProRes encoder does NOT propagate -color_primaries / -color_trc into the stream: ffprobe
# reports them as "unknown" and only the matrix survives, so Resolve would not know the file is
# PQ Rec.2020 and would interpret it as SDR. The prores_metadata bitstream filter writes them
# into the bitstream properly. Caught by verifying the output rather than trusting the flags.
PRORES_TAG = ['-bsf:v', 'prores_metadata=color_primaries=bt2020:color_trc=smpte2084:'
                        'colorspace=bt2020nc']

CODECS = {
    'prores4444':   {'args': ['-c:v', 'prores_ks', '-profile:v', '4', '-vendor', 'apl0',
                              '-bits_per_mb', '8000', '-pix_fmt', 'yuv444p10le'] + PRORES_TAG,
                     'ext': 'mov', 'label': 'Apple ProRes 4444, 4:4:4'},
    'prores4444xq': {'args': ['-c:v', 'prores_ks', '-profile:v', '5', '-vendor', 'apl0',
                              '-bits_per_mb', '16000', '-pix_fmt', 'yuv444p10le'] + PRORES_TAG,
                     'ext': 'mov', 'label': 'Apple ProRes 4444 XQ, 4:4:4'},
    # DNxHR cannot be tagged correctly through this muxer: without +write_colr it comes out
    # claiming bt709, and with it the matrix is right but primaries and transfer stay "unknown".
    # +write_colr is the lesser evil, and the limitation is reported rather than hidden.
    'dnxhr444':     {'args': ['-c:v', 'dnxhd', '-profile:v', 'dnxhr_444',
                              '-pix_fmt', 'yuv444p10le'],
                     'mux': ['-movflags', '+write_colr'],
                     'ext': 'mov', 'label': 'Avid DNxHR 444, 4:4:4',
                     'tagging': 'INCOMPLETE: matrix bt2020nc only; primaries and transfer are '
                                'written as unknown. Set the clip to Rec.2020 / ST2084 by hand '
                                'in Resolve, or prefer prores4444, which tags fully.'},
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
           *spec.get('mux', []),
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
               'bit_depth_note': 'ProRes 4444 is a 12-bit container and ffprobe reports '
                                 'yuv444p12le, but ffmpeg accepts only 10-bit input, so the '
                                 'samples carry 10 bits. For more, take '
                                 'intermediate/linear_4k/*.exr into Resolve.',
               'colour_tagging': {
                   'prores': 'prores_metadata bitstream filter, and NO +write_colr (it '
                             'downgrades the colr atom to a legacy form that cannot express PQ)',
                   'dnxhr': 'cannot be fully tagged through this muxer; primaries and transfer '
                            'stay unknown. Prefer prores4444.',
                   'verified_by': 'probing the written file with ffprobe, not by trusting flags'},
               'exports': results}
    shot.log('master_export.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
