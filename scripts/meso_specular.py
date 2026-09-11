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


def meso_modulate(E_k, n, l, mask, sigma, wrap=0.0, eps=0.02, mod_max=4.0, strength=1.0):
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
    mod = np.clip((c_np + eps) / (lo + eps), 0.0, mod_max)
    # `strength` is an exponent, so it scales the relief in STOPS and leaves 1 (no change)
    # fixed: 0 is the proxy alone, 1 is the neural normals in full.
    if strength != 1.0:
        mod = np.clip(mod ** strength, 0.0, mod_max)
    mod = mod.astype(np.float32)
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


def kajiya_kay_specular(E_k, t, n, l, v, dist, radius, exponent, f0, azimuthal=2.0):
    """Anisotropic fibre highlight from source k, in the same units as ggx_specular.

    Round 45. Hair is not a surface with a normal, it is a bundle of cylinders: a fibre reflects
    a source into a CONE about its own axis, which is why a head of hair carries a band that
    runs ACROSS the strands and moves when the light moves rather than when the head turns. The
    lobe used here is Kajiya-Kay in its half-vector form -- `sin(T, H) ** p`, maximal where the
    half-vector is perpendicular to the fibre, which is exactly the reflection-cone condition --
    with a Schlick Fresnel on the cuticle and the Blinn-Phong normalisation `(p + 2) / (8 pi)`.

    Two honest caveats, because this is not a fibre BCSDF. The normalisation constant is a
    SURFACE lobe's, borrowed: a fibre's lobe is one-dimensional, so the constant is an
    approximation and the magnitude is cross-checked against the room's own measured band rather
    than asserted (composite_ratio logs `hair_specular.fr_peak_model` beside
    `fr_peak_measured`). And this is the primary (R) highlight only -- Marschner's TT and the
    coloured TRT, and the cuticle tilt that shifts them, are not modelled; on near-black hair
    the R lobe carries nearly all of the visible sheen because so little light returns from
    inside the fibre.

    `exponent` and `radius` combine the way the GGX lobe's roughness and source size do: the
    lobe's half-width in `sin(T, H)` units is widened by half the source's angular radius, so a
    60 cm panel at 0.5 m gives a broad sheen and a 4 cm bulb a tight one from one exponent.

    `azimuthal` is not decoration, it is what makes this a BAND. Kajiya-Kay has no azimuthal term
    at all: `sin(T, H)` is 1 for EVERY tangent lying in the surface whenever the half-vector is
    near the surface normal, so a front-lit, front-viewed head glows all over -- measured here on
    the first build, where the added highlight hit its clamp on the median hair pixel and the
    crown read as a lit wig. A real fibre is a cylinder, and only the part of it whose own normal
    faces the half-vector reflects: `max(n . h, 0) ** azimuthal` on the surface normal is the
    cheap stand-in for that (Marschner's azimuthal term done properly needs the fibre's offset
    from the cylinder axis, which a plate cannot give). 0 restores textbook Kajiya-Kay.
    """
    t_ = _t()
    tt, nn, ll, vv = _dev(t), _dev(n), _dev(l), _dev(v)
    d = _dev(dist)
    h = ll + vv
    h = h / t_.clamp(t_.linalg.norm(h, dim=1, keepdim=True), min=1e-6)
    th = t_.clamp(t_.sum(tt * h, dim=1, keepdim=True), min=-1.0, max=1.0)
    sin_th = t_.sqrt(t_.clamp(1.0 - th * th, min=0.0))
    # Half-width of sin^p about the peak, widened by the source's angular radius, back to p.
    p0 = t_.as_tensor(float(exponent), device=th.device, dtype=th.dtype)
    w = t_.sqrt(2.0 * np.log(2.0) / t_.clamp(p0, min=1.0))
    w = w + radius / (2.0 * t_.clamp(d, min=1e-3))
    p = t_.clamp(2.0 * np.log(2.0) / t_.clamp(w * w, min=1e-9), min=1.0)
    lobe = t_.pow(t_.clamp(sin_th, min=1e-6), p)
    vdh = t_.clamp(t_.sum(vv * h, dim=1, keepdim=True), min=0.0)
    F = _dev(f0)
    F = F + (1.0 - F) * (1.0 - vdh) ** 5
    fr = F * (p + 2.0) / (8.0 * np.pi) * lobe
    if azimuthal > 0:
        ndh = t_.clamp(t_.sum(nn * h, dim=1, keepdim=True), min=0.0)
        fr = fr * t_.pow(ndh, azimuthal)
    L = fr * np.pi * _dev(E_k)
    return gpu_image.to_cpu(L), gpu_image.to_cpu(fr, squeeze=True)
