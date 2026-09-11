"""Round 44: meso-scale shading from the NEURAL normals, and a physically scaled specular,
per rendered light source. torch on CUDA; numpy in, numpy out, one device round trip each.

Why (rounds/round43.md SS3). The ratio composite multiplies the photograph by
E_target / E_source. E_source is the old light evaluated on the neural normals at full
resolution, so the division removes the capture room's meso-shading; E_target is the proxy
render, which carries 40-60 % of the relief that a Lambert of the same normals under the new
light produces (the shell is a stride-2 grid, smooth-shaded, under 3.7 mm subsurface). Two
lights then describe her form at once -- the softbox's in what the division left, the bulb's
at half strength -- and the additive specular was normalised to peak 1 and multiplied by
Fresnel 0.04, so it never exceeded 0.03 anywhere. That combination is the "plastic skin".

MESO-SHADING. For each source k with its own rendered irradiance E_k (bulb, fill: linear
transport, E_k sum to the full pass) and world position, the neural normals' cosine
c = max(n . l_k, 0) with l_k per PIXEL (a bulb 0.5 m from a 20 cm face is not a distant
light) is split at a scale sigma: E_k' = E_k x c / lowpass_sigma(c). Above sigma the proxy's
field stands (occlusion, falloff, the source's solid angle); below it the neural normals'
relief replaces the proxy's. sigma is MEASURED per shot as the scale where the proxy stops
resolving what the normals resolve (the same crossover test composite_ratio.detail_sigma
applies to E_source), never a pixel count typed in.

SPECULAR. f_r = F(v.h) D(n.h) V(n.l, n.v) -- GGX with Smith height-correlated visibility and
Schlick Fresnel -- times the source's irradiance at the pixel, pi x E_k' (the ratio pass is a
0.6 albedo Lambertian, so E_k = E / pi in radiance units). The source's angular size widens
the lobe: alpha' = alpha + r / (2 d) (Karis 2013, a half-vector spread of half the angular
radius), which is what makes a 60 cm panel at 0.5 m a broad soft sheen and a 4 cm bulb a hot
spot, from the same roughness. The magnitude is not a setting: with roughness 0.30 the peak
of F D is 0.04 / (pi 0.09^2) ~ 1.6 x E, about ten times a skin diffuse of rho E / pi.
"""
import numpy as np
import gpu_image

LUMA = np.array([.2126, .7152, .0722], np.float32)


def _t():
    return gpu_image._torch()


def _dev(a):
    return gpu_image.to_dev(a)


def masked_blur(x, m, sigma):
    """Gaussian low-pass of x normalised by the blurred mask, so the silhouette does not bleed."""
    if sigma <= 0:
        return x
    num = gpu_image.gaussian_blur(x * m[..., None] if x.ndim == 3 else x * m, sigma)
    den = gpu_image.gaussian_blur(m, sigma)
    den = np.maximum(den, 1e-4)
    return num / (den[..., None] if x.ndim == 3 else den)


def crossover_sigma(E_lum, c, mask, scales=(2, 4, 8, 16, 32, 64)):
    """Scale at which the proxy's rendered field stops resolving the neural Lambert.

    Relative RMS of a log high-pass of each field over the mask, per scale; the crossover is
    where the proxy's relief first equals the Lambert's. Same construction as
    composite_ratio.detail_sigma, applied to the NEW light. Returns (sigma_px, table).
    """
    import cv2
    if not mask.any():
        return 0.0, []
    le = np.log2(np.maximum(E_lum, 1e-6))
    lc = np.log2(np.maximum(c, 1e-3))
    table, prev, cross = [], None, 0.0
    for sc in scales:
        a = float((le - cv2.GaussianBlur(le, (0, 0), sc))[mask].std())
        b = float((lc - cv2.GaussianBlur(lc, (0, 0), sc))[mask].std())
        ratio = b / max(a, 1e-9)
        table.append({'scale_px': sc, 'proxy_rms': round(a, 4), 'lambert_rms': round(b, 4),
                      'lambert_over_proxy': round(ratio, 3)})
        if prev is not None and prev[1] > 1.0 >= ratio and cross == 0.0:
            s0, r0 = prev
            t = (r0 - 1.0) / max(r0 - ratio, 1e-9)
            cross = float(np.exp(np.log(s0) + t * (np.log(sc) - np.log(s0))))
        prev = (sc, ratio)
    if cross == 0.0 and table and table[-1]['lambert_over_proxy'] > 1.0:
        cross = float(scales[-1])
    return cross, table


def meso_modulate(E_k, n, l, mask, sigma, wrap=0.0, eps=0.02, mod_max=4.0):
    """E_k' = E_k x c / lowpass(c): the neural normals' cosine relief below `sigma`.

    `wrap` widens the cosine for a large source (a 60 cm panel is not a point): c = clamp((n.l +
    wrap) / (1 + wrap)). `eps` keeps the ratio finite past the terminator, where both c and its
    low-pass vanish and the proxy's own zero decides. The modulation is clamped at `mod_max`
    (a high-pass ratio, so 1 is "no change"); the low end is free -- a cheek rolling off a hard
    side light is the point. Returns (E_k', modulation).
    """
    t = _t()
    nn, ll, mm = _dev(n), _dev(l), _dev(mask.astype(np.float32))
    c = t.clamp((t.sum(nn * ll, dim=1, keepdim=True) + wrap) / (1.0 + wrap), min=0.0)
    c = c * mm
    c_np = gpu_image.to_cpu(c, squeeze=True)
    lo = masked_blur(c_np, mask.astype(np.float32), sigma)
    mod = np.clip((c_np + eps) / (lo + eps), 0.0, mod_max).astype(np.float32)
    mod = np.where(mask, mod, 1.0).astype(np.float32)
    return E_k * mod[..., None], mod


def ggx_specular(E_k, n, l, v, dist, radius, rough, f0):
    """Specular radiance from source k, in the same units as E_k (scene radiance / anchor).

    E_k must already carry the source's occlusion and cosine (it is the RENDERED irradiance,
    E / pi); pi x E_k is the irradiance the lobe is driven by. rough and f0 are per-pixel
    fields; radius is the source's radius in metres (a panel: the disc of equal area).
    """
    t = _t()
    nn, ll, vv = _dev(n), _dev(l), _dev(v)
    d = _dev(dist)
    h = ll + vv
    h = h / t.clamp(t.linalg.norm(h, dim=1, keepdim=True), min=1e-6)
    ndl = t.clamp(t.sum(nn * ll, dim=1, keepdim=True), min=0.0)
    ndv = t.clamp(t.sum(nn * vv, dim=1, keepdim=True), min=1e-4)
    ndh = t.clamp(t.sum(nn * h, dim=1, keepdim=True), min=0.0)
    vdh = t.clamp(t.sum(vv * h, dim=1, keepdim=True), min=0.0)
    a = t.clamp(_dev(rough), min=1e-3) ** 2
    # Source size widens the lobe: half-vector spread is half the source's angular radius.
    a = t.clamp(a + radius / (2.0 * t.clamp(d, min=1e-3)), max=1.0)
    a2 = a * a
    den = ndh * ndh * (a2 - 1.0) + 1.0
    D = a2 / t.clamp(np.pi * den * den, min=1e-8)
    F = _dev(f0)
    F = F + (1.0 - F) * (1.0 - vdh) ** 5
    # Smith height-correlated visibility, V = G / (4 n.l n.v).
    gv = ndl * t.sqrt(ndv * ndv * (1.0 - a2) + a2)
    gl = ndv * t.sqrt(ndl * ndl * (1.0 - a2) + a2)
    V = 0.5 / t.clamp(gv + gl, min=1e-6)
    fr = F * D * V
    L = fr * np.pi * _dev(E_k)
    return gpu_image.to_cpu(L), gpu_image.to_cpu(fr, squeeze=True)
