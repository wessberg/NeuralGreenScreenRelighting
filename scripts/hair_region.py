"""Round 45: where the HAIR is, and which way its strands run -- both read from the plate.

The hair is the one surface in this shot that the pipeline has no mask for. The skin mask
covers skin, the face oval covers the face, the matte covers everything; the hair is what is
left of the head after the skin comes out. Nothing here is tuned to this performer: the region
is defined in units of her own face (the mediapipe face oval's radius), and the strand
direction is measured from the image by a structure tensor, so a different haircut, colour or
parting changes the numbers and not the code.

REGION. `hair_mask` = solid matte AND outside the skin mask AND inside `radius` face-radii of
the face oval's centroid AND not below the chin by more than `below_chin` face-radii. The last
two cuts are what keep a dark jacket and a shadowed shoulder out; without them "everything that
is not skin" is most of the frame. The crown -- the part the artist named -- is the top
`crown_frac` of that mask's height, which is where a light hanging above the head lands.

STRANDS. Hair is a strongly oriented texture, so the structure tensor of the log luminance has
one large and one small eigenvalue over it, and the small one's eigenvector runs ALONG the
strands. Two details matter. It is computed on the LOG, so a bright band does not weight the
orientation more than dark hair beside it; and the tensor is averaged over a window before the
eigenvector is taken (averaging the gradients instead would cancel opposite gradients across a
single strand and return nothing). `coherence` = (l1 - l2) / (l1 + l2) is 0 on an unoriented
patch and 1 on a perfect grating -- it is the honest confidence of the tangent, and every stage
downstream weights by it rather than assuming the direction is good everywhere.

The 2D tangent is lifted to world axes by the plate camera's basis (image +x is world +X, image
+y is world -Z, the camera looks down +Y) and then projected onto the surface's tangent plane
with the neural normal, which is the direction a hair fibre actually lies in.
"""
import numpy as np
import cv2


def face_frame(pts, topology):
    """(centroid, radius, chin_y) of the face oval, in pixels. The unit every cut below uses."""
    oval = pts[topology['face_oval']][:, :2].astype(np.float32)
    c = oval.mean(0)
    r = float(np.linalg.norm(oval - c, axis=1).max())
    return c, r, float(oval[:, 1].max())


def hair_mask(alpha, skin, pts, topology, radius=1.55, below_chin=0.15, alpha_min=0.5,
              skin_max=0.5, close_px=5, oval_pad=0.07, skin_grow=0.025, close_scalp=0.08,
              parts=False):
    """Solid matte, off the skin, OUTSIDE the face oval, inside `radius` face-radii of the centre.

    `skin` is 0..1 at plate resolution. The face-oval cut is not redundant with the skin mask:
    the skin mask excludes the eyes, brows, nostrils and lips, so "matte and not skin" hands
    back every one of them as if it were hair (measured -- the first version of this probe
    flagged her irises, lashes and lower lip as crown sheen). `oval_pad` widens the oval by that
    fraction of a face radius, because mediapipe's oval crosses just under the brows.
    `skin_grow` dilates the skin mask by that fraction of a face radius before subtracting it:
    the hairline is a mixed transition and the skin mask stops INSIDE it, so without the grow a
    rim of lit forehead comes back as "hair" -- measured at 37 % of the >4x band's pixels and a
    THIRD of its energy on this plate, all of it at skin chroma rather than the room's. A few
    pixels of real wispy hair go with it, which is the right trade for a stage that subtracts
    light. `close_px` closes the lash-thin gaps the skin mask leaves along the hairline.

    `alpha_min` is deliberately LOW. The crown's brightest sheen sits on the silhouette arc,
    where the matte is fractional; at alpha > 0.99 the region stopped just inside it and the arc
    kept its shine while everything below went dark -- the artist saw that immediately ("it
    still shines along the arc/edge of her head"). The caller weights by alpha instead.

    `close_scalp` puts her PARTING back. The parting is scalp, so the skin mask claims it, and
    subtracting the skin mask cut a hard strip down the middle of the crown that no sheen stage
    could touch -- leaving it bright over a darkened crown, which the artist read at once as "a
    gap like she's bald". Scalp that shows THROUGH hair is enclosed by hair on both sides, so a
    morphological closing at this fraction of a face radius bridges it while leaving the hairline
    (hair on one side only) exactly where it is.

    With `parts=True` returns (mask, cuts), where `cuts` is the same region WITHOUT the matte
    term. Feather the weight on `cuts` -- those edges (hairline, face oval, radius) are the
    arbitrary ones -- and let alpha carry the silhouette, which is not arbitrary at all.
    """
    h, w = alpha.shape
    c, r, chin_y = face_frame(pts, topology)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot(xx - c[0], yy - c[1])
    oval = pts[topology['face_oval']][:, :2].astype(np.float32)
    padded = c + (oval - c) * (1.0 + oval_pad)
    face = np.zeros((h, w), np.uint8)
    cv2.fillPoly(face, [np.round(padded).astype(np.int32)], 255)
    sk = (skin >= skin_max).astype(np.uint8)
    g = int(round(skin_grow * r))
    if g > 0:
        sk = cv2.dilate(sk, np.ones((2 * g + 1, 2 * g + 1), np.uint8))
    outer = (face == 0) & (d < radius * r) & (yy < chin_y + below_chin * r)
    cuts = (sk == 0) & outer
    m = cuts & (alpha > alpha_min)
    if close_px > 0:
        k = np.ones((close_px, close_px), np.uint8)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
        m &= cuts & (alpha > alpha_min)
    cs = int(round(close_scalp * r))
    if cs > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cs + 1, 2 * cs + 1))
        enclosed = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
        cuts = cuts | (enclosed & outer)
        m = m | (enclosed & outer & (alpha > alpha_min))
    return (m, cuts) if parts else m


def crown_mask(hair, frac=0.45):
    """The top `frac` of the hair region's vertical extent -- where an overhead light lands."""
    ys = np.where(hair.any(1))[0]
    if len(ys) == 0:
        return np.zeros_like(hair)
    y0, y1 = int(ys.min()), int(ys.max())
    cut = y0 + frac * (y1 - y0)
    yy = np.arange(hair.shape[0])[:, None]
    return hair & (yy <= cut)


def structure_tangent(lum, mask, grad_sigma=1.0, tensor_sigma=4.0):
    """(tangent_2d, coherence): the strand direction and how oriented the texture is.

    Returns a unit (h, w, 2) field in image axes (x right, y down) and a 0..1 coherence. The
    tensor is J = blur(grad grad^T) of the log luminance; the tangent is the eigenvector of the
    SMALLER eigenvalue (the across-strand gradient is the large one). Every blur goes through
    gpu_image, so the whole field is computed on the GPU when one is present.
    """
    import gpu_image
    l = np.log2(np.maximum(lum, 1e-4)).astype(np.float32)
    # Outside the region the log of near-black hair against a black matte is a cliff, and the
    # tensor window reaches across it; a blurred stand-in keeps the silhouette from printing a
    # spurious "strand direction" along the outline.
    l = np.where(mask, l, gpu_image.gaussian_blur(l, 8.0))
    ls = gpu_image.gaussian_blur(l, grad_sigma)
    gx = np.zeros_like(ls); gy = np.zeros_like(ls)
    gx[:, 1:-1] = (ls[:, 2:] - ls[:, :-2]) * 0.5
    gy[1:-1, :] = (ls[2:, :] - ls[:-2, :]) * 0.5
    j = gpu_image.gaussian_blur(np.stack([gx * gx, gy * gy, gx * gy], -1), tensor_sigma)
    jxx, jyy, jxy = j[..., 0], j[..., 1], j[..., 2]
    tr = jxx + jyy
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - (jxx * jyy - jxy * jxy), 0.0))
    l1 = tr / 2.0 + disc                       # across the strands
    l2 = tr / 2.0 - disc                       # along them
    coh = (l1 - l2) / np.maximum(l1 + l2, 1e-12)
    # Eigenvector of l1 is (jxy, l1 - jxx); the tangent is perpendicular to it.
    ex, ey = jxy, l1 - jxx
    n = np.hypot(ex, ey)
    degenerate = n < 1e-9
    n = np.maximum(n, 1e-12)
    t2 = np.stack([-(ey / n), ex / n], -1).astype(np.float32)   # rotate 90 deg -> along strands
    t2[degenerate] = np.array([1.0, 0.0], np.float32)
    return t2, coh.astype(np.float32)


def tangent_world(t2, nrm):
    """Lift the image tangent into world axes and project it onto the surface.

    Plate camera: +X right, +Y away from camera, +Z up (stress_proxy.py), so an image direction
    (dx, dy) with y DOWN is the world direction (dx, 0, -dy). The fibre lies in the surface, so
    the world tangent is that direction with its normal component removed. Where the projection
    collapses (the lift is parallel to the normal, i.e. the surface turns edge-on to the strand
    direction) the field falls back to the residual of world +X, and the caller should lean on
    `coherence` there rather than on the direction.
    """
    d = np.stack([t2[..., 0], np.zeros_like(t2[..., 0]), -t2[..., 1]], -1)
    t = d - nrm * np.sum(d * nrm, -1, keepdims=True)
    n = np.linalg.norm(t, axis=-1, keepdims=True)
    bad = (n[..., 0] < 1e-3)
    if bad.any():
        alt = np.array([1.0, 0.0, 0.0], np.float32) - nrm * nrm[..., 0:1]
        an = np.maximum(np.linalg.norm(alt, axis=-1, keepdims=True), 1e-6)
        t = np.where(bad[..., None], alt / an, t / np.maximum(n, 1e-6))
    else:
        t = t / np.maximum(n, 1e-6)
    return t.astype(np.float32)
