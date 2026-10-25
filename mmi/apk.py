"""A-priori-knowledge (APK) classification of reconstructed images, after Eqs. (2)-(5)
of the IPIGS paper (Liu, Gong, Chen, EMBC 2025).

An input image is labelled 'confused' when a stroke-like bright spot is found
 (Eq. 2)  hue/value test        F_C(H,V): hue in the hemorrhage hue range AND value > V
 (Eq. 3)  location test         F_L(C,L): the spot centre lies in a dielectric-sensitive area
                                          Part 1 (thalamus disc) or Part 2 (arc just inside the skull)
 (Eq. 4)  size test             F_S(R)  : equivalent radius within the preset stroke radius
 (Eq. 5)  F = F_C AND F_L AND F_S
and 'clear' otherwise.  The two classes train two separate Pix2pix weight sets.

Images are rendered with the paper's 0-70 colour scale (viridis, yellow = high eps').
HSV follows the OpenCV convention (H in 0..179, V in 0..255): hue 20-30, V > 193.
"""
import numpy as np
from scipy import ndimage as ndi
import matplotlib
from matplotlib import colors as mcolors

from .phantom import THALAMUS, BONE

HUE_RANGE = (20, 30)
V_MIN = 193
VMAX_EPS = 70.0


def render_rgb(eps_img, vmin=0.0, vmax=VMAX_EPS):
    cmap = matplotlib.colormaps["viridis"]
    return (cmap(np.clip((eps_img - vmin) / (vmax - vmin), 0, 1))[..., :3] * 255).astype(np.uint8)


def rgb_to_hsv_cv(rgb):
    hsv = mcolors.rgb_to_hsv(rgb / 255.0)
    return hsv[..., 0] * 179.0, hsv[..., 2] * 255.0


def candidate_mask(eps_img):
    """Eq. (2): pixels whose colour is in the preset hemorrhage hue range and bright."""
    H, V = rgb_to_hsv_cv(render_rgb(eps_img))
    return (H >= HUE_RANGE[0]) & (H <= HUE_RANGE[1]) & (V > V_MIN)


def sensitive_areas(scene, grid, thal_radius_mm=15.0, skull_band_mm=(5.0, 15.0)):
    """Part 1 (thalamus disc) and Part 2 (band just inside the inner skull surface) on `grid`."""
    ids = scene._grid_ids(grid)
    h_mm = grid.h * 1e3
    thal = np.isin(ids, THALAMUS)
    if thal.sum() > 0:
        c = ndi.center_of_mass(thal)
        yy, xx = np.indices(ids.shape)
        part1 = np.hypot(yy - c[0], xx - c[1]) * h_mm <= thal_radius_mm
    else:
        part1 = np.zeros(ids.shape, dtype=bool)
    bone = np.isin(ids, BONE)
    inside = ndi.binary_fill_holes(bone) & ~bone
    dist = ndi.distance_transform_edt(inside) * h_mm        # mm from skull inner surface
    part2 = inside & (dist >= skull_band_mm[0]) & (dist <= skull_band_mm[1])
    return part1, part2


def classify(eps_img, scene, grid, radius_mm=(4.0, 14.0), min_px=3):
    """Return ('confused'|'clear', info dict). Eq. (5): F_C AND F_L AND F_S."""
    cand = candidate_mask(eps_img)
    part1, part2 = sensitive_areas(scene, grid)
    h_mm = grid.h * 1e3
    lab, n = ndi.label(cand)
    info = dict(n_candidates=int(n), hits=[])
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() < min_px:
            continue
        cy, cx = ndi.center_of_mass(m)
        r_mm = np.sqrt(m.sum() / np.pi) * h_mm
        in_loc = bool(part1[int(round(cy)), int(round(cx))] or part2[int(round(cy)), int(round(cx))])
        in_size = radius_mm[0] <= r_mm <= radius_mm[1]
        info["hits"].append(dict(centre_px=(cx, cy), radius_mm=r_mm, location=in_loc, size=in_size))
        if in_loc and in_size:
            return "confused", info
    return "clear", info
