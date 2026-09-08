"""Panasonic V-Log / V-Gamut -> scene-linear Rec.709, and the checks that keep it honest.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/vlog.py --selftest

Every stage of this pipeline assumes scene-linear Rec.709: the keyer thresholds, the depth and
normal estimators, the SH irradiance ratio, the ratio composite, the PQ grade. V-Log is a log
curve over a wide gamut, so feeding it in undecoded makes all of them wrong *while still
producing a picture* -- lifted midtones, crushed saturation, and every measured number in
`rounds/` silently invalid. There is no error to catch; that is what makes it dangerous.

**The constants here are TRANSCRIBED, not measured, and that distinction is the point.**
They match Panasonic's published V-Log specification, and `--selftest` checks the properties that
a correct transcription must have:

* 18% mid grey encodes to about 0.423, the documented anchor;
* the curve is continuous at the cut point, so no seam appears in a gradient;
* encode and decode round-trip;
* every row of the V-Gamut -> Rec.709 matrix sums to 1.0, which is what makes white stay white.

Those checks catch a typo. They do NOT prove the constants are right for this camera. Only a shot
reference does that: put a ColorChecker and an 18%% grey card under known light at the head of a
take, and confirm the grey card lands where the curve says and the patch colours match the chart.
Until that has been done, treat any number derived from his footage as provisional -- and note
that a wrong decode would bias the whole pipeline in a direction that looks like a grading choice
rather than a bug.

Never apply a creative display LUT first. It is not invertible, and it discards the highlight
range that makes the HDR grade worth doing.
"""
import sys, json, argparse
import numpy as np

# Panasonic V-Log, as published.
CUT1, CUT2 = 0.01, 0.181
B, C, D = 0.00873, 0.241514, 0.598206

# V-Gamut -> Rec.709. Rows sum to 1.0, which is the white-preserving property --selftest checks.
VGAMUT_TO_REC709 = np.array([
    [1.806576, -0.695697, -0.110879],
    [-0.170090, 1.305955, -0.135865],
    [-0.025206, -0.154468, 1.179674],
], dtype=np.float32)
REC709_TO_VGAMUT = np.linalg.inv(VGAMUT_TO_REC709).astype(np.float32)

MID_GREY_LINEAR = 0.18
MID_GREY_VLOG_DOCUMENTED = 0.423


def encode(linear):
    """Scene-linear reflectance -> V-Log code value in [0,1]."""
    x = np.asarray(linear, np.float32)
    return np.where(x < CUT1, 5.6 * x + 0.125, C * np.log10(np.maximum(x + B, 1e-10)) + D)


def decode(code):
    """V-Log code value in [0,1] -> scene-linear reflectance."""
    v = np.asarray(code, np.float32)
    return np.where(v < CUT2, (v - 0.125) / 5.6, np.power(10.0, (v - D) / C) - B)


def to_linear_rec709(vlog_vgamut):
    """The whole ingest transform: V-Log code over V-Gamut -> scene-linear Rec.709.

    Order matters and is not interchangeable: linearise FIRST, then apply the gamut matrix. A
    matrix is a linear operation and applying it to log-encoded values mixes channels that are
    not in a linear relationship, which produces plausible-looking and wrong colour.
    """
    lin = decode(vlog_vgamut)
    return lin @ VGAMUT_TO_REC709.T


def from_linear_rec709(linear_rec709):
    """Inverse, for round-trip tests and for synthesising test material."""
    return encode(linear_rec709 @ REC709_TO_VGAMUT.T)


def selftest():
    checks = []

    def chk(name, ok, detail):
        checks.append({'test': name, 'pass': bool(ok), 'detail': detail})

    g = float(encode(MID_GREY_LINEAR))
    chk('18% grey encodes to the documented anchor',
        abs(g - MID_GREY_VLOG_DOCUMENTED) < 0.005,
        'encode(0.18) = %.4f, documented %.3f' % (g, MID_GREY_VLOG_DOCUMENTED))

    lo, hi = float(encode(CUT1 - 1e-6)), float(encode(CUT1 + 1e-6))
    chk('curve is continuous at the cut point', abs(lo - hi) < 1e-3,
        'encode just below %.6f vs just above %.6f, step %.2e' % (lo, hi, abs(lo - hi)))

    x = np.array([0.0, 0.005, 0.02, 0.18, 0.5, 1.0, 4.0, 16.0], np.float32)
    err = float(np.max(np.abs(decode(encode(x)) - x)))
    chk('encode/decode round-trips', err < 1e-4, 'max abs error %.3e over 0..16' % err)

    rs = VGAMUT_TO_REC709.sum(1)
    chk('V-Gamut matrix rows sum to 1 (white stays white)',
        np.allclose(rs, 1.0, atol=2e-6), 'row sums %s' % np.array2string(rs, precision=6))

    w = np.array([[1.0, 1.0, 1.0]], np.float32) @ VGAMUT_TO_REC709.T
    chk('neutral maps to neutral', np.allclose(w, 1.0, atol=2e-6),
        'V-Gamut white -> Rec.709 %s' % np.array2string(w[0], precision=6))

    inv = float(np.max(np.abs(VGAMUT_TO_REC709 @ REC709_TO_VGAMUT - np.eye(3))))
    chk('matrix inverse is consistent', inv < 1e-5, 'max |M M^-1 - I| = %.2e' % inv)

    rgb = np.random.RandomState(0).rand(64, 64, 3).astype(np.float32) * 2.0
    rt = float(np.max(np.abs(to_linear_rec709(from_linear_rec709(rgb)) - rgb)))
    chk('full ingest transform round-trips', rt < 1e-3, 'max abs error %.3e' % rt)

    ok = all(c['pass'] for c in checks)
    report = {
        'constants': {'cut1': CUT1, 'cut2': CUT2, 'b': B, 'c': C, 'd': D,
                      'vgamut_to_rec709': VGAMUT_TO_REC709.tolist()},
        'status': 'TRANSCRIBED from the published V-Log specification, NOT measured for this '
                  'camera body',
        'checks': checks,
        'verdict': 'PASS' if ok else 'FAIL',
        'what_this_does_not_prove': (
            'These checks catch a transcription error. They cannot confirm the curve matches this '
            'camera. Shoot a ColorChecker and an 18%% grey card under known light at the head of a '
            'take: the grey card must land at %.3f and the decoded patches must match the chart. '
            'Until then, every number measured from his footage is provisional.'
            % MID_GREY_VLOG_DOCUMENTED),
    }
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else print(__doc__) or 0)
