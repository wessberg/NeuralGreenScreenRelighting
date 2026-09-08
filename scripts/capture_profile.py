"""Describe any capture format once, and derive everything the pipeline needs from it.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/capture_profile.py --list
    ... --profile s5iix_6k_open_gate --delivery 3840x1600 --squeeze 1.6

A shot is defined by more than its pixel count, and every one of these axes changes what an
earlier stage must do:

* **anamorphic or spherical** -- decides whether the plate must be desqueezed before any
  estimator sees it, and what focal length the CG camera gets;
* **sensor aspect** (3:2 open gate, 17:9, 16:9) -- decides how much has to be cropped to reach
  the delivery aspect, and therefore the real field of view;
* **chroma subsampling** (4:4:4, 4:2:2, 4:2:0) -- decides whether chroma must be upsampled before
  keying, because a green-screen key is a chroma operation;
* **GOP structure** (all-intra vs Long GOP) -- decides whether an intermediate transcode is
  required before keying, because inter-frame prediction puts mosquito noise exactly on the
  silhouette and makes the matte crawl;
* **transfer and gamut** (V-Log/V-Gamut, Rec.709, log variants) -- decides what has to be decoded
  on ingest. Everything downstream assumes scene-linear Rec.709;
* **frame rate** -- decides whether a conform is needed, which is a creative decision and must be
  recorded rather than defaulted.

The point is that none of these should be rediscovered per shot or hardcoded in a stage. A shot
names a profile; stages ask this module what to do. `requirements()` returns the ordered list of
ingest actions a given profile implies, so a stage cannot silently skip one.

Profiles are data, not code: `capture_format.json` holds the artist's own rig, and anything here
is a starting point that a shot may override. Nothing in this module measures anything -- it
computes consequences from stated facts, and a stated fact that is wrong stays wrong.
"""
import sys, json, math, argparse
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Full-frame reference gate. A profile that uses a smaller readout states its own fraction.
FULL_FRAME_MM = (36.0, 24.0)

CHROMA = {
    '4:4:4': {'h': 1, 'v': 1, 'upsample_before_key': False},
    '4:2:2': {'h': 2, 'v': 1, 'upsample_before_key': True},
    '4:2:0': {'h': 2, 'v': 2, 'upsample_before_key': True},
}

TRANSFERS = {
    # name: (needs_decode_to_linear, note)
    'rec709': (True, 'Standard gamma; srgb_to_linear is close enough and is what the reporter '
                     'plate uses. Not the same as sRGB but within the tolerance of this pipeline.'),
    'srgb': (True, 'Straight sRGB EOTF.'),
    'v-log': (True, 'Panasonic V-Log. Constants MUST be verified against Panasonic documentation '
                    'and a shot ColorChecker; they are not implemented yet.'),
    'slog3': (True, 'Sony S-Log3. Not implemented.'),
    'linear': (False, 'Already scene-linear; nothing to do.'),
}

GAMUTS = {
    'rec709': (False, 'Already the working space.'),
    'v-gamut': (True, 'Panasonic V-Gamut -> Rec.709 matrix required. Not implemented yet.'),
    'sgamut3.cine': (True, 'Sony. Not implemented.'),
    'rec2020': (True, 'Matrix required.'),
}

PROFILES = {
    's5iix_6k_open_gate': {
        'camera': 'Panasonic LUMIX S5 IIX',
        'label': '6K 3:2 open gate, HEVC 10-bit 4:2:0 Long GOP 200 Mbps',
        'resolution': [5952, 3968],
        'sensor_fraction': [1.0, 1.0],
        'bit_depth': 10, 'chroma': '4:2:0', 'intra': False,
        'container': 'MOV', 'codec': 'HEVC', 'bitrate_mbps': 200,
        'transfer': 'v-log', 'gamut': 'v-gamut',
        'frame_rate': '24/1',
        'note': 'Open gate is Long GOP / MOV only on this body: an all-intra transcode before '
                'keying is mandatory, not optional.',
    },
    's5iix_58k_17_9': {
        'camera': 'Panasonic LUMIX S5 IIX',
        'label': '5.8K 17:9, ProRes 422 HQ 1.6 Gbps all-intra, LPCM',
        'resolution': [5776, 3056],
        'sensor_fraction': [5776 / 5952, 3056 / 3968],
        'bit_depth': 10, 'chroma': '4:2:2', 'intra': True,
        'container': 'MOV', 'codec': 'ProRes 422 HQ', 'bitrate_mbps': 1600,
        'transfer': 'v-log', 'gamut': 'v-gamut',
        'frame_rate': '24/1',
        'note': 'All-intra and 4:2:2: the two properties that most help a green-screen key.',
    },
    'stock_1080p_rec709': {
        'camera': 'unknown (stock footage)',
        'label': 'Reporter test plate: 1080p, Rec.709, spherical',
        'resolution': [1920, 1080],
        'sensor_fraction': [1.0, 1.0],
        'bit_depth': 8, 'chroma': '4:2:0', 'intra': False,
        'container': 'MP4', 'codec': 'H.264', 'bitrate_mbps': None,
        'transfer': 'rec709', 'gamut': 'rec709',
        'frame_rate': '60000/1001',
        'anamorphic': False,
        'note': 'Not the artist\'s footage. Carries its own camera block in shot.json and must '
                'never inherit capture_format.json.',
    },
}


def derive(profile, delivery=(3840, 1600), squeeze=1.6, taking_focal_mm=35.0,
           gate_mm=FULL_FRAME_MM):
    """Everything a stage needs, computed from the profile plus the delivery frame."""
    p = dict(profile)
    w, h = p['resolution']
    dw, dh = delivery
    anamorphic = p.get('anamorphic', True) and squeeze != 1.0
    sq = squeeze if anamorphic else 1.0

    recorded_aspect = w / h
    desqueezed_aspect = recorded_aspect * sq
    delivery_aspect = dw / dh

    # Crop whichever axis is in surplus, never scale non-uniformly.
    if desqueezed_aspect > delivery_aspect:
        keep_w, keep_h = delivery_aspect / desqueezed_aspect, 1.0
    else:
        keep_w, keep_h = 1.0, desqueezed_aspect / delivery_aspect
    used_w, used_h = w * keep_w, h * keep_h
    desq_w = used_w * sq
    oversampling = desq_w / dw

    sensor_w = gate_mm[0] * p['sensor_fraction'][0] * keep_w
    sensor_h = gate_mm[1] * p['sensor_fraction'][1] * keep_h
    render_focal = taking_focal_mm / sq
    hfov = 2 * math.degrees(math.atan(sensor_w / 2 / render_focal))
    vfov = 2 * math.degrees(math.atan(sensor_h / 2 / taking_focal_mm))

    chroma = CHROMA[p['chroma']]
    tr_decode, tr_note = TRANSFERS[p['transfer']]
    gm_convert, gm_note = GAMUTS[p['gamut']]

    return {
        'profile': p.get('label'), 'camera': p.get('camera'),
        'anamorphic': anamorphic, 'squeeze': sq,
        'recorded': {'resolution': [w, h], 'aspect': round(recorded_aspect, 4),
                     'chroma': p['chroma'], 'bit_depth': p['bit_depth'],
                     'intra': p['intra'], 'codec': p.get('codec'),
                     'frame_rate': p['frame_rate']},
        'desqueezed_aspect': round(desqueezed_aspect, 4),
        'delivery': {'resolution': list(delivery), 'aspect': round(delivery_aspect, 4)},
        'crop': {'keep_width_pct': round(100 * keep_w, 2),
                 'keep_height_pct': round(100 * keep_h, 2),
                 'used_pixels': [round(used_w), round(used_h)],
                 'discarded_pct': round(100 * (1 - keep_w * keep_h), 2)},
        'oversampling_linear': round(oversampling, 3),
        'camera_model': {'taking_focal_mm': taking_focal_mm,
                         'render_focal_mm': round(render_focal, 4),
                         'sensor_used_mm': [round(sensor_w, 2), round(sensor_h, 2)],
                         'hfov_deg': round(hfov, 2), 'vfov_deg': round(vfov, 2)},
        'colour': {'transfer': p['transfer'], 'needs_transfer_decode': tr_decode,
                   'transfer_note': tr_note,
                   'gamut': p['gamut'], 'needs_gamut_convert': gm_convert,
                   'gamut_note': gm_note},
        'chroma_detail': chroma,
        'note': p.get('note'),
    }


def requirements(d):
    """Ordered ingest actions this profile implies. A stage must not silently skip one."""
    out = []
    if d['colour']['needs_transfer_decode']:
        out.append({'step': 'decode_transfer', 'blocking': True,
                    'why': 'Every stage downstream assumes scene-linear. %s'
                           % d['colour']['transfer_note'],
                    'detail': 'transfer=%s' % d['colour']['transfer']})
    if d['colour']['needs_gamut_convert']:
        out.append({'step': 'convert_gamut', 'blocking': True,
                    'why': 'Working space is Rec.709. %s' % d['colour']['gamut_note'],
                    'detail': 'gamut=%s' % d['colour']['gamut']})
    if not d['recorded']['intra']:
        out.append({'step': 'transcode_to_all_intra', 'blocking': True,
                    'why': 'Long GOP puts mosquito noise on high-contrast edges -- the '
                           'silhouette and every hair -- so the matte crawls frame to frame.',
                    'detail': 'ProRes 4444 or DNxHR 444 intermediate, before keying'})
    if d['chroma_detail']['upsample_before_key']:
        out.append({'step': 'upsample_chroma', 'blocking': True,
                    'why': 'A green-screen key is a chroma operation, and chroma is stored at '
                           '1/%d horizontally and 1/%d vertically.'
                           % (d['chroma_detail']['h'], d['chroma_detail']['v']),
                    'detail': 'interpolate to full resolution before keying, not nearest'})
    if d['anamorphic']:
        out.append({'step': 'desqueeze', 'blocking': True,
                    'why': 'Depth and normal estimators are trained on spherical imagery; a '
                           'squeezed face is %.2fx too narrow and yields geometry that is '
                           'plausible and wrong.' % d['squeeze'],
                    'detail': 'desqueeze %.3fx horizontally before any estimator' % d['squeeze']})
    if d['crop']['discarded_pct'] > 0.01:
        out.append({'step': 'crop_to_delivery_aspect', 'blocking': False,
                    'why': 'Recorded aspect does not match delivery; %.1f%% is discarded and the '
                           'effective field of view narrows accordingly.'
                           % d['crop']['discarded_pct'],
                    'detail': 'keep %.1f%% of width, %.1f%% of height'
                              % (d['crop']['keep_width_pct'], d['crop']['keep_height_pct'])})
    if d['oversampling_linear'] > 1.05:
        out.append({'step': 'key_before_downscale', 'blocking': False,
                    'why': '%.2fx linear oversampling over delivery. A matte computed at native '
                           'resolution and downscaled afterwards is effectively supersampled, so '
                           'hair edges get real fractional alpha instead of a hard decision.'
                           % d['oversampling_linear'],
                    'detail': 'key at native, downscale after'})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--profile', default=None)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--delivery', default='3840x1600')
    ap.add_argument('--squeeze', type=float, default=1.6)
    ap.add_argument('--focal', type=float, default=35.0)
    ap.add_argument('--compare', action='store_true', help='derive every profile side by side')
    args = ap.parse_args()

    if args.list or (not args.profile and not args.compare):
        print(json.dumps({k: v['label'] for k, v in PROFILES.items()}, indent=2))
        return
    dw, dh = (int(x) for x in args.delivery.lower().split('x'))

    names = list(PROFILES) if args.compare else [args.profile]
    out = {}
    for n in names:
        if n not in PROFILES:
            raise SystemExit('Unknown profile %r. Known: %s' % (n, ', '.join(PROFILES)))
        sq = args.squeeze if PROFILES[n].get('anamorphic', True) else 1.0
        d = derive(PROFILES[n], (dw, dh), sq, args.focal)
        d['ingest_requirements'] = requirements(d)
        out[n] = d
    print(json.dumps(out if len(out) > 1 else out[names[0]], indent=2))


if __name__ == '__main__':
    main()
